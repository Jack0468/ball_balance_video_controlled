module master(

	
    );
	 
	localparam [3:0] IDLE      = 4'd0,
						  FETCH     = 4'd1,
						  START     = 4'd2,
						  TRANSMIT  = 4'd3,
						  STOP      = 4'd4,
						  INC_ROM   = 4'd5,
					
	 
	wire config_adr; //0x42
	 
	reg [1:0]  state = FETCH;       // The FSM memory (starts in FETCH)
	reg [23:0] shift_reg;           // The 24-bit "bucket brigade" for our data
	reg        sda_drive = 1'b1;    // The internal register controlling the SDA pin

	// I2C requires Open-Drain outputs. 
	// If sda_drive is 1, we let the wire float high (1'bz). 
	// If sda_drive is 0, we pull the wire low to 0V (1'b0).
	assign SDA = sda_drive ? 1'bz : 1'b0;
	 
	 ov7607_setup_rom ROM(.clk(), .address(), .data( ... ));
	 
	 // main FSM
	 
	 always @(posedge i2c_clk or negedge reset) begin
		case (state)
			IDLE:
			
				scl <= 1'b1;
				sda <= 1'b1;
			
			FETCH: begin
				shift_reg <= {8'h42, rom_data[15:0]}; // address and 16 bit word
				sda <= 1'b1;
				
				state <= START;
			end 
			
			START: begin
				sda <= 1'b0; //drive sda low
				
				bit_counter <= 5'd27;
				
				state <= TRANSMIT;
			end 
			
			TRANSMIT: begin 
				
				


endmodule
