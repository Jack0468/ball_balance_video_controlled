`timescale 1ns / 1ps

module tb_sdram_arbiter;

    // Clocks
    reg clk_100mhz;
    initial begin
        clk_100mhz = 0;
        forever #5 clk_100mhz = ~clk_100mhz; // 100MHz
    end

    reg pclk;
    initial begin
        #3 pclk = 0;
        forever #20 pclk = ~pclk; // 25MHz
    end

    reg ti_clk;
    initial begin
        #2 ti_clk = 0;
        forever #10 ti_clk = ~ti_clk; // 50MHz
    end

    // Reset signals
    reg rst_n = 1;
    reg frame_rst = 0;

    // Camera Write Interface
    reg cam_wr_en = 0;
    reg [15:0] cam_wr_data = 0;

    // USB Read Interface
    reg usb_rd_en = 0;
    wire [15:0] usb_rd_data;
    wire usb_empty;

    // SDRAM Interface (dummy)
    wire [12:0] sdram_a;
    wire [ 1:0] sdram_ba;
    wire [15:0] sdram_dq;
    wire        sdram_cke;
    wire        sdram_cs_n;
    wire        sdram_ras_n;
    wire        sdram_cas_n;
    wire        sdram_we_n;
    wire [ 1:0] sdram_dqm;

    wire init_complete;

    // Instantiate Arbiter
    sdram_arbiter uut (
        .clk_100mhz(clk_100mhz),
        .rst_n(rst_n),
        .frame_rst(frame_rst),

        .pclk(pclk),
        .cam_wr_en(cam_wr_en),
        .cam_wr_data(cam_wr_data),

        .ti_clk(ti_clk),
        .usb_rd_en(usb_rd_en),
        .usb_rd_data(usb_rd_data),
        .usb_empty(usb_empty),

        .sdram_a(sdram_a),
        .sdram_ba(sdram_ba),
        .sdram_dq(sdram_dq),
        .sdram_cke(sdram_cke),
        .sdram_cs_n(sdram_cs_n),
        .sdram_ras_n(sdram_ras_n),
        .sdram_cas_n(sdram_cas_n),
        .sdram_we_n(sdram_we_n),
        .sdram_dqm(sdram_dqm),

        .init_complete(init_complete)
    );

    // =========================================================
    // Minimal SDRAM BFM (Synchronous, CAS Latency 2)
    // =========================================================
    reg [15:0] sdram_mem [0:1048575];
    
    reg [12:0] row_latch = 0;
    reg [1:0]  ba_latch  = 0;
    
    reg [1:0] read_pipe = 0;
    reg [21:0] read_addr = 0;
    
    always @(posedge clk_100mhz) begin
        // Latch row and bank on ACTIVE command (ras_n=0, cas_n=1)
        if (!sdram_cs_n && !sdram_ras_n && sdram_cas_n) begin
            row_latch <= sdram_a;
            ba_latch  <= sdram_ba;
        end
        
        // Handle WRITE command (cas_n=0, we_n=0)
        if (!sdram_cs_n && !sdram_cas_n && !sdram_we_n) begin
            sdram_mem[{ba_latch, row_latch[7:0], sdram_a[9:0]}] <= sdram_dq;
        end
        
        // Handle READ command (cas_n=0, we_n=1) -> CAS latency 2
        // Shift register: bit 0 goes high 1 cycle after READ command.
        read_pipe <= {read_pipe[0], (!sdram_cs_n && !sdram_cas_n && sdram_we_n)};
        if (!sdram_cs_n && !sdram_cas_n && sdram_we_n) begin
            read_addr <= {ba_latch, row_latch[7:0], sdram_a[9:0]};
        end
    end
    
    // Drive data continuously based on the pipeline. 
    // Data appears combinatorially AFTER the first posedge, 
    // so it is valid and ready to be sampled ON the second posedge (CAS Latency 2).
    assign sdram_dq = (read_pipe[0]) ? sdram_mem[read_addr] : 16'bz;

    // =========================================================
    // Test sequences
    // =========================================================
    // Simulate a smaller chunk (e.g. 1024 pixels) instead of a full frame
    // so the simulation runs in seconds instead of minutes.
    integer num_pixels = 1024; // 1024 pixels
    integer i, j;
    integer err_count = 0;
    reg [15:0] expected_val;

    initial begin
        $dumpfile("tb_sdram_arbiter.vcd");
        $dumpvars(0, tb_sdram_arbiter);
        
        $display("Starting SDRAM Arbiter Simulation...");
        
        // 1. Reset
        rst_n = 0;
        frame_rst = 1;
        #100;
        rst_n = 1;
        frame_rst = 0;

        // 2. Wait for init complete
        $display("Waiting for SDRAM init_complete...");
        @(posedge init_complete);
        $display("SDRAM initialized successfully at %0t ns", $time);

        #1000; // Wait a bit after init

        // 3. Start Camera Write phase
        $display("Writing %0d pixels to arbiter...", num_pixels);
        @(posedge pclk);
        for (i = 0; i < num_pixels; i = i + 1) begin
            cam_wr_en = 1;
            cam_wr_data = i[15:0];
            @(posedge pclk);
            
            // camera_read.v only outputs one pixel every 2 pclk cycles.
            // We must simulate this, otherwise we feed data at 25MHz 
            // which overflows the SDRAM write bandwidth (~14MHz).
            cam_wr_en = 0;
            @(posedge pclk);
        end
        cam_wr_en = 0;

        $display("Finished writing to arbiter. Waiting for SDRAM writes to complete...");
        
        // Let SDRAM writes flush
        #2000; 

        // 4. Start USB Read phase
        $display("Reading pixels from arbiter via USB...");
        for (i = 0; i < num_pixels; i = i + 1) begin
            
            // Wait for not empty
            while (usb_empty) begin
                @(posedge ti_clk);
            end

            usb_rd_en = 1;
            @(posedge ti_clk);
            usb_rd_en = 0;
            @(posedge ti_clk); // data valid 1 cycle after en
            
            expected_val = i[15:0];
            if (usb_rd_data !== expected_val) begin
                $display("ERROR at pixel %0d: Expected %0h, Got %0h", i, expected_val, usb_rd_data);
                err_count = err_count + 1;
                if (err_count > 10) begin
                    $display("Too many errors. Aborting.");
                    $finish;
                end
            end
        end

        if (err_count == 0) begin
            $display("ALL TESTS PASSED! (%0d pixels successfully read and written)", num_pixels);
        end else begin
            $display("FAILED with %0d errors", err_count);
        end

        $finish;
    end

endmodule
