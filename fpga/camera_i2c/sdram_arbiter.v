`timescale 1ns / 1ps

module sdram_arbiter (
    input  wire        clk_100mhz,  // SDRAM Clock
    input  wire        rst_n,       // Active low reset for SDRAM controller
    input  wire        frame_rst,   // Active high reset for pointers and FIFOs

    // Camera Write Interface (runs on pclk)
    input  wire        pclk,
    input  wire        cam_wr_en,
    input  wire [15:0] cam_wr_data,

    // USB Read Interface (runs on ti_clk)
    input  wire        ti_clk,
    input  wire        usb_rd_en,
    output wire [15:0] usb_rd_data,
    output wire        usb_empty,

    // SDRAM Physical Pins
    output wire [12:0] sdram_a,
    output wire [ 1:0] sdram_ba,
    inout  wire [15:0] sdram_dq,
    output wire        sdram_cke,
    output wire        sdram_cs_n,
    output wire        sdram_ras_n,
    output wire        sdram_cas_n,
    output wire        sdram_we_n,
    output wire [ 1:0] sdram_dqm,
    
    // Status
    output wire        init_complete
);

    // =========================================================================
    // CDC Synchronizers for 100MHz domain
    // =========================================================================
    reg [1:0] frame_rst_sync = 0;
    always @(posedge clk_100mhz) frame_rst_sync <= {frame_rst_sync[0], frame_rst};
    wire frame_rst_100 = frame_rst_sync[1];

    reg [1:0] rst_n_sync = 2'b11; // Active low, default: not in reset
    always @(posedge clk_100mhz) rst_n_sync <= {rst_n_sync[0], rst_n};
    wire rst_n_100 = rst_n_sync[1];

    // =========================================================================
    // Camera Write FIFO (Crosses pclk -> clk_100mhz)
    // =========================================================================
    wire        cam_fifo_empty;
    wire [15:0] cam_fifo_dout;
    reg         cam_fifo_rd_en;
    
    fifo write_fifo (
        .rst    (frame_rst),
        .wr_clk (pclk),
        .wr_en  (cam_wr_en),
        .din    (cam_wr_data),
        .full   (), // SDRAM is effectively infinite for a single frame
        
        .rd_clk (clk_100mhz),
        .rd_en  (cam_fifo_rd_en),
        .dout   (cam_fifo_dout),
        .empty  (cam_fifo_empty)
    );

    // =========================================================================
    // USB Read FIFO (Crosses clk_100mhz -> ti_clk)
    // =========================================================================
    wire        usb_fifo_full;
    wire        usb_fifo_wr_en;
    wire [15:0] usb_fifo_din;
    
    fifo read_fifo (
        .rst    (frame_rst),
        .wr_clk (clk_100mhz),
        .wr_en  (usb_fifo_wr_en),
        .din    (usb_fifo_din),
        .full   (usb_fifo_full),
        
        .rd_clk (ti_clk),
        .rd_en  (usb_rd_en),
        .dout   (usb_rd_data),
        .empty  (usb_empty)
    );

    // =========================================================================
    // SDRAM Controller Instantiation
    // =========================================================================
    wire        sdram_busy;
    
    reg  [23:0] sdr_wr_addr_cmd;
    reg  [15:0] sdr_wr_data_cmd;
    reg         sdr_wr_en_cmd;
    
    reg  [23:0] sdr_rd_addr_cmd;
    wire [15:0] sdr_rd_data_out;
    wire        sdr_rd_ready;
    reg         sdr_rd_en_cmd;
    
    sdram_controller sdram_ctrl (
        .wr_addr   (sdr_wr_addr_cmd),
        .wr_data   (sdr_wr_data_cmd),
        .wr_enable (sdr_wr_en_cmd),

        .rd_addr   (sdr_rd_addr_cmd),
        .rd_data   (sdr_rd_data_out),
        .rd_ready  (sdr_rd_ready),
        .rd_enable (sdr_rd_en_cmd),

        .busy      (sdram_busy),
        .rst_n     (rst_n_100),
        .clk       (clk_100mhz),

        .addr          (sdram_a),
        .bank_addr     (sdram_ba),
        .data          (sdram_dq),
        .clock_enable  (sdram_cke),
        .cs_n          (sdram_cs_n),
        .ras_n         (sdram_ras_n),
        .cas_n         (sdram_cas_n),
        .we_n          (sdram_we_n),
        .data_mask_low (sdram_dqm[0]),
        .data_mask_high(sdram_dqm[1])
    );

    // =========================================================================
    // SDRAM Initialization Tracking
    // =========================================================================
    // The SDRAM controller runs an init sequence after rst_n deasserts.
    // With the fixed busy signal, busy stays HIGH throughout init.
    // We latch init_done on the first busy LOW after reset.
    reg sdram_init_done = 0;
    always @(posedge clk_100mhz) begin
        if (!rst_n_100)
            sdram_init_done <= 1'b0;
        else if (!sdram_busy && !sdram_init_done)
            sdram_init_done <= 1'b1;  // First idle after init = init complete
    end

    assign init_complete = sdram_init_done;

    // =========================================================================
    // SDRAM Pointers
    // =========================================================================
    reg [23:0] sdram_write_ptr = 0;
    reg [23:0] sdram_read_ptr  = 0;
    
    // =========================================================================
    // Arbitration State Machine
    // =========================================================================
    //
    // Design principles:
    //   1. Never issue a command when sdram_busy is HIGH
    //   2. Never issue a command before init is complete
    //   3. After issuing wr/rd_enable, wait for busy to go HIGH (acceptance)
    //      then wait for busy to go LOW (completion)
    //   4. Writes have priority over reads (camera data is real-time)
    //
    localparam STATE_IDLE        = 3'd0,
               STATE_WRITE_FETCH = 3'd1,  // Wait 1 cycle for FIFO read latency
               STATE_WRITE_CMD   = 3'd2,  // Present write data + assert wr_enable
               STATE_WRITE_WAIT  = 3'd3,  // Wait for busy HIGH then LOW
               STATE_READ_CMD    = 3'd4,  // Present read address + assert rd_enable
               STATE_READ_WAIT   = 3'd5;  // Wait for busy HIGH then LOW
               
    reg [2:0] state = STATE_IDLE;

    // Track whether the controller accepted our command (busy went HIGH)
    reg cmd_accepted = 0;
    
    always @(posedge clk_100mhz) begin
        if (frame_rst_100) begin
            // Frame reset: zero pointers, stop all commands, return to idle.
            // Does NOT re-trigger SDRAM init (only rst_n does that).
            sdram_write_ptr <= 0;
            sdram_read_ptr  <= 0;
            sdr_wr_en_cmd   <= 0;
            sdr_rd_en_cmd   <= 0;
            cam_fifo_rd_en  <= 0;
            cmd_accepted    <= 0;
            state           <= STATE_IDLE;
        end else begin
            // Default: deassert single-cycle strobes
            sdr_wr_en_cmd  <= 0;
            sdr_rd_en_cmd  <= 0;
            cam_fifo_rd_en <= 0;
            
            case (state)
                // ---------------------------------------------------------
                // IDLE: Check for pending work. Writes have priority.
                // Gate on: init done AND controller not busy.
                // ---------------------------------------------------------
                STATE_IDLE: begin
                    cmd_accepted <= 0;
                    if (sdram_init_done && !sdram_busy) begin
                        if (!cam_fifo_empty) begin
                            // Camera data waiting — start write sequence
                            cam_fifo_rd_en <= 1'b1;
                            state <= STATE_WRITE_FETCH;
                        end else if (!usb_fifo_full && (sdram_read_ptr < sdram_write_ptr)) begin
                            // SDRAM has data to send to USB — start read sequence
                            sdr_rd_addr_cmd <= sdram_read_ptr;
                            sdr_rd_en_cmd   <= 1'b1;
                            state <= STATE_READ_CMD;
                        end
                    end
                end

                // ---------------------------------------------------------
                // WRITE: Fetch data from FIFO (1-cycle read latency)
                // ---------------------------------------------------------
                STATE_WRITE_FETCH: begin
                    state <= STATE_WRITE_CMD;
                end
                
                // ---------------------------------------------------------
                // WRITE: Present data to SDRAM controller
                // ---------------------------------------------------------
                STATE_WRITE_CMD: begin
                    sdr_wr_addr_cmd <= sdram_write_ptr;
                    sdr_wr_data_cmd <= cam_fifo_dout;
                    sdr_wr_en_cmd   <= 1'b1;
                    sdram_write_ptr <= sdram_write_ptr + 1'b1;
                    cmd_accepted    <= 0;
                    state <= STATE_WRITE_WAIT;
                end
                
                // ---------------------------------------------------------
                // WRITE: Wait for controller to accept (busy HIGH) then 
                //        complete (busy LOW). With the fixed busy signal,
                //        busy goes HIGH on the cycle after wr_enable and
                //        stays HIGH until the write sequence finishes.
                // ---------------------------------------------------------
                STATE_WRITE_WAIT: begin
                    if (sdram_busy)
                        cmd_accepted <= 1'b1;
                    
                    if (cmd_accepted && !sdram_busy)
                        state <= STATE_IDLE;
                end

                // ---------------------------------------------------------
                // READ: Hold rd_enable for 1 cycle (already asserted in IDLE),
                //       then wait for acceptance + completion.
                // ---------------------------------------------------------
                STATE_READ_CMD: begin
                    cmd_accepted <= 0;
                    state <= STATE_READ_WAIT;
                end
                
                // ---------------------------------------------------------
                // READ: Wait for busy HIGH (acceptance) then LOW (completion).
                //       rd_ready will pulse when data is available.
                // ---------------------------------------------------------
                STATE_READ_WAIT: begin
                    if (sdram_busy)
                        cmd_accepted <= 1'b1;
                    
                    if (cmd_accepted && !sdram_busy) begin
                        sdram_read_ptr <= sdram_read_ptr + 1'b1;
                        state <= STATE_IDLE;
                    end
                end
            endcase
        end
    end
    
    // =========================================================================
    // Read data path: SDRAM rd_ready -> USB read FIFO
    // =========================================================================
    assign usb_fifo_wr_en = sdr_rd_ready;
    assign usb_fifo_din   = sdr_rd_data_out;

endmodule
