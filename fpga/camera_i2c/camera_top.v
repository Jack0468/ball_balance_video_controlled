module camera_top
    (
        // from Opal Kelly
        input  wire [7:0]  hi_in,
        output wire [1:0]  hi_out,
        inout  wire [15:0] hi_inout,
        output wire        hi_muxsel,
        inout  wire        i2c_sda,
        inout  wire        i2c_scl,
	    input wire         clk1,
        input wire         clk2,

        // from OV7670
        input wire         pclk,
        input wire         vsync,
        input wire         href,
        input wire [7:0]   p_data,
	
        // to OV7670
        output wire        v_sup,
        output wire        xclk,
	    inout  wire        siod,
        output wire        sioc,

        // SDRAM Physical Pins
        output wire [12:0] sdram_a,
        output wire [ 1:0] sdram_ba,
        inout  wire [15:0] sdram_dq,
        output wire        sdram_cke,
        output wire        sdram_cs_n,
        output wire        sdram_ras_n,
        output wire        sdram_cas_n,
        output wire        sdram_we_n,
        output wire [ 1:0] sdram_dqm
    );

	//target interface bus
	wire ti_clk;
	wire [30:0] ok1;
	wire [16:0] ok2;
	wire [17*8-1:0] ok2x;
	
	//pipe out 
	wire pipe_out_read;
	wire pipe_out_ready;
	wire [15:0] pipe_out_data;

	// from python 
	wire [15:0] start; 
	wire [15:0] reset;
	wire [15:0] arm_wire;

	// camera read
    wire done;
    wire [15:0] pixel_data;
    wire pixel_valid;
    wire frame_done;
	 wire frame_start;
	 
	// SDRAM arbiter status
	 wire empty;
	
	 
	 assign hi_muxsel = 1'b0;   // connect FPGA to the USB microcontroller, not the PROM
	assign i2c_sda   = 1'bz;   // release the I2C bus as it's irrelevant 
	assign i2c_scl   = 1'bz;

    assign xclk = clk2;  // clk from fpga to camera


    // xclk 
    camera_config #(.CLK_FREQ(100000000)) writer(
        .clk(clk1),
        .start(start[0]),
        .sioc(sioc),
        .siod(siod),
        .done(done)
        );
    
    //pclk from camera
    camera_read reader(
		  .config_done(done),
		  .full(1'b0),  // SDRAM acts as infinite buffer; never block camera_read
        .p_clock(pclk),
        .vsync(vsync),
        .href(href),
        .p_data(p_data),
        .pixel_data(pixel_data),
        .pixel_valid(pixel_valid),
        .frame_done(frame_done),
		  .frame_start(frame_start)
    );



	// 0. Hardwire camera PWDN low to keep it awake
	assign v_sup = 1'b0;

	// 1. Detect Host Arm Request
	
	reg [2:0] arm_s = 0;
	always @(posedge pclk) arm_s <= {arm_s[1:0], arm_wire[0]};
	wire arm_rise = arm_s[1] & ~arm_s[2];

	// 2. Safe FIFO Reset State Machine
	localparam S_IDLE       = 3'd0;
	localparam S_WAIT_VSYNC = 3'd1;
	localparam S_RST        = 3'd2;
	localparam S_RECOVERY   = 3'd3;
	localparam S_CAP        = 3'd4;

	reg [2:0] state = S_IDLE;
	reg [3:0] rst_cnt = 0;
	reg armed = 0;
	reg fifo_rst_r = 0;

	// Dedicated frame capture completion flag for Python polling (WO_FRAME).
	// Unlike the old 'full' signal which was HIGH in S_IDLE (both before AND
	// after capture), this only goes HIGH when a capture actually completes.
	reg frame_captured = 0;

	always @(posedge pclk) begin
		case (state)
			S_IDLE: begin
				armed <= 0; 
				fifo_rst_r <= 0;
				if (arm_rise) begin
					frame_captured <= 0;  // Clear on new ARM request
					state <= S_WAIT_VSYNC;
				end
			end
			
			S_WAIT_VSYNC: begin
				// Wait for VSYNC to drop, signaling the start of a frame
				if (frame_start) begin 
					rst_cnt <= 0;
					state <= S_RST; 
				end
			end
			
			S_RST: begin
				// Hold FIFO reset high for 10 clock cycles safely
				fifo_rst_r <= 1;
				rst_cnt <= rst_cnt + 1'b1;
				if (rst_cnt == 4'd10) begin
					fifo_rst_r <= 0;
					rst_cnt <= 0;
					state <= S_RECOVERY;
				end
			end
			
			S_RECOVERY: begin
				// Wait 10 clock cycles for FIFO to safely recover before writing
				rst_cnt <= rst_cnt + 1'b1;
				if (rst_cnt == 4'd10) begin
					armed <= 1; 
					state <= S_CAP;
				end
			end
			
			S_CAP: begin
				if (frame_done) begin 
					armed <= 0;
					frame_captured <= 1;  // Signal to Python: frame is in SDRAM
					state <= S_IDLE; 
				end
			end
		endcase
	end

	// 3. Connect the Reset
	reg [1:0] rst_s = 0;
	always @(posedge pclk) rst_s <= {rst_s[0], reset[0]};

	// Reset is active if Python manually requests it, OR if the state machine is resetting
	wire fifo_rst = rst_s[1] | fifo_rst_r;

	// 4. Overflow detection (for debugging via WireOut)
	reg overflow_sticky = 0;
	always @(posedge pclk) begin
		if (frame_start) overflow_sticky <= 0;
		// Note: with SDRAM buffering, overflow is extremely unlikely
	end

		// SDRAM Arbiter (Replacing old 4KB FIFO)
	wire sdram_init_complete;

	sdram_arbiter arbiter (
		.clk_100mhz(clk1),
		.rst_n(~reset[0]), // Active low reset from host
		.frame_rst(fifo_rst_r), // Active high reset from arm signal

		.pclk(pclk),
		.cam_wr_en(pixel_valid & armed),
		.cam_wr_data(pixel_data),

		.ti_clk(ti_clk),
		.usb_rd_en(pipe_out_read),
		.usb_rd_data(pipe_out_data),
		.usb_empty(empty), // Used by WireOut to check if read is valid if needed

		.sdram_a(sdram_a),
		.sdram_ba(sdram_ba),
		.sdram_dq(sdram_dq),
		.sdram_cke(sdram_cke),
		.sdram_cs_n(sdram_cs_n),
		.sdram_ras_n(sdram_ras_n),
		.sdram_cas_n(sdram_cas_n),
		.sdram_we_n(sdram_we_n),
		.sdram_dqm(sdram_dqm),

		.init_complete(sdram_init_complete)
	);		
	 
	 assign pipe_out_ready = ~empty;
	 
	 //debugging 
	reg [15:0] pclk_cnt  = 0;
	reg [15:0] vsync_cnt = 0;
	reg [15:0] href_cnt  = 0;
	always @(posedge pclk) begin
		pclk_cnt <= pclk_cnt + 1'b1;                 
		if (frame_start) vsync_cnt <= vsync_cnt + 1; 
		if (href)        href_cnt  <= href_cnt  + 1; 
	end
	  
	 //okHost interface with endpoints 
	okHost okHI(
	.hi_in(hi_in), .hi_out(hi_out), .hi_inout(hi_inout), .ti_clk(ti_clk),
	.ok1(ok1), .ok2(ok2));
	
	okWireOR # (.N(8)) wireOR (ok2, ok2x);

	okWireIn  wi02(.ok1(ok1),.ep_addr(8'h02), .ep_dataout(start));
	okWireIn  wi03(.ok1(ok1), .ep_addr(8'h03), .ep_dataout(reset));
	okWireIn  wi04(.ok1(ok1), .ep_addr(8'h04), .ep_dataout(arm_wire));

	okWireOut wo20(.ok1(ok1), .ok2(ok2x[6*17 +: 17]), .ep_addr(8'h20),
					   .ep_datain({15'd0, done}));

	// WO_FRAME: Python polls this to know when a frame is ready in SDRAM
	okWireOut wo21(.ok1(ok1), .ok2(ok2x[0*17 +: 17]), .ep_addr(8'h21),
						.ep_datain({14'd0, sdram_init_complete, frame_captured}));
						
	okWireOut wo22(.ok1(ok1), .ok2(ok2x[1*17 +: 17]), .ep_addr(8'h22),
					   .ep_datain({15'd0, empty}));

	okWireOut wo23(.ok1(ok1), .ok2(ok2x[2*17 +: 17]), .ep_addr(8'h23),
					   .ep_datain({15'd0, overflow_sticky}));
	
	okWireOut wo24(.ok1(ok1), .ok2(ok2x[3*17 +: 17]), .ep_addr(8'h24),
					   .ep_datain(vsync_cnt));
						
	okWireOut wo25(.ok1(ok1), .ok2(ok2x[4*17 +: 17]), .ep_addr(8'h25),
					   .ep_datain(href_cnt));
	
	okWireOut wo26(.ok1(ok1), .ok2(ok2x[5*17 +: 17]), .ep_addr(8'h26),
					   .ep_datain(pclk_cnt));
	


	okPipeOut epA0(.ok1(ok1), .ok2(ok2x[7*17 +: 17]), .ep_addr(8'ha0), 
						  .ep_read(pipe_out_read), 
						  .ep_datain(pipe_out_data));

						  

endmodule

