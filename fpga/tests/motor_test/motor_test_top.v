
//jp3_29 -> enable
//jp3_37 -> step
//jp3_39 -> dir 
//jp3_43 -> 3V 

module motor_test_top(
	input  wire [7:0]  hi_in,
   output wire [1:0]  hi_out,
   inout  wire [15:0] hi_inout,
   output wire        hi_muxsel,
   inout  wire        i2c_sda,
   inout  wire        i2c_scl,
	
	input wire clk1,
	output reg step,
	output wire enable,
	output wire dir,
	output wire volt_supply,
	output wire test1,
	output wire test2,
	output wire test3
   );
	

	assign hi_muxsel = 1'b0;   // connect FPGA to the USB microcontroller, not the PROM
	assign i2c_sda   = 1'bz;   // release the I2C bus so we don't fight the USB micro
	assign i2c_scl   = 1'bz;

	wire [30:0] ok1;
	wire [16:0] ok2;
	wire [16:0] ok2x;

	// from python 
	wire [15:0] value_in; // N
	wire [15:0] reset;

	okHost okHI(
		.hi_in(hi_in), .hi_out(hi_out), .hi_inout(hi_inout), .ti_clk(ti_clk),
		.ok1(ok1), .ok2(ok2));

	okWireOR # (.N(1)) wireOR (ok2, ok2x);

	okWireIn  wi00(.ok1(ok1),.ep_addr(8'h00), .ep_dataout(value_in));
	okWireIn  wi01(.ok1(ok1), .ep_addr(8'h01), .ep_dataout(reset)); 

	reg [15:0] counter;
	reg [15:0] prescale;
	reg        tick;
	
	always @(posedge clk1) begin
		prescale <= prescale + 1'b1;
		tick     <= (prescale == 16'd0);   // 1-cycle pulse every 2^15 clk1
	end
	
	// based on slower 'tick', this steps the motor
	always@(posedge tick) begin
		
		if (reset[0]) begin 
			counter <= 16'd0;
			step <= 1'd0;
		end else if (counter >= value_in) begin 
			step <= ~step;
			counter <= 16'd0; 
		end else begin
			counter <= counter + 16'd1;
		end
	end 
	
	// assign other values
	assign enable = 1'b0;
	assign dir = 1'b1; 
	assign volt_supply = 1'b1;
	
	assign test1 = 1'b1;
	assign test2 = 1'b1;
	assign test3 = 1'b1;
	
	okWireOut wo20 (.ok1(ok1), .ok2(ok2x), .ep_addr(8'h20),
		.ep_datain({step, counter[14:0]}));

endmodule