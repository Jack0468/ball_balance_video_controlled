
module camera_read(
	input wire config_done,
	input wire full,
	input wire p_clock,
	input wire vsync,
	input wire href,
	input wire [7:0] p_data,
	output reg [15:0] pixel_data =0,
	output reg pixel_valid = 0,
	output wire frame_done,
	output wire frame_start
   );
	 
	
	reg [1:0] FSM_state = 2;
    reg pixel_half = 0;
	
	localparam WAIT_FRAME_START = 0;
	localparam ROW_CAPTURE = 1;
	localparam WAIT_CONFIG = 2;

	reg r1_vsync, r2_vsync;

	// edges of vsync
	initial {r1_vsync, r2_vsync} = 0;
	always @(posedge p_clock) 
		{r2_vsync, r1_vsync} <= {r1_vsync, vsync};
	
	assign frame_start = (r1_vsync == 0) && (r2_vsync == 1);  // Negative Edge of vsync
   assign frame_done  = (r1_vsync == 1) && (r2_vsync == 0);  // Positive Edge of vsync 
	

	//AGAIN NEW STUFF THAT IS BROKEN!!!

	// two stage flip-flop synchroniser for config_done to avoid metastability 
	reg [1:0] r_config_done = 0;
	always @(posedge p_clock) begin
		r_config_done <= {r_config_done[0], config_done};
	end
	
	wire f_config_done = r_config_done[1];
	 
	always@(posedge p_clock)
	begin 
		case(FSM_state)
			WAIT_CONFIG: begin 
				FSM_state <= f_config_done ? WAIT_FRAME_START : WAIT_CONFIG;
			end

			WAIT_FRAME_START: begin //wait for VSYNC
				FSM_state <= (frame_start) ? ROW_CAPTURE : WAIT_FRAME_START;
				pixel_half <= 0;
			end
			
			ROW_CAPTURE: begin 
				FSM_state <= frame_done ? WAIT_FRAME_START : ROW_CAPTURE;
				pixel_valid <= (href && pixel_half) ? 1 : 0; 
				if (href) begin
					pixel_half <= ~ pixel_half;
					if (pixel_half) pixel_data[7:0] <= p_data;
					else pixel_data[15:8] <= p_data;
				end else begin
				    pixel_half <= 0;
				end
			end
		
		endcase
	end
endmodule
