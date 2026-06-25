// top.v  -  XEM3010 / FrontPanel host interface
// Reads WireIn 0x00 and returns value+1 on WireOut 0x20.

module top(
   input  wire [7:0]  hi_in,
   output wire [1:0]  hi_out,
   inout  wire [15:0] hi_inout,
   output wire        hi_muxsel,
   inout  wire        i2c_sda,
   inout  wire        i2c_scl
   );

// XEM3010 housekeeping (required for FrontPanel comms):
assign hi_muxsel = 1'b0;   // connect FPGA to the USB microcontroller, not the PROM
assign i2c_sda   = 1'bz;   // release the I2C bus so we don't fight the USB micro
assign i2c_scl   = 1'bz;

// Target interface bus from okHost
wire        ti_clk;
wire [30:0] ok1;
wire [16:0] ok2;
wire [16:0] ok2x;


wire [15:0] value_in;

okHost okHI(
   .hi_in(hi_in), .hi_out(hi_out), .hi_inout(hi_inout), .ti_clk(ti_clk),
   .ok1(ok1), .ok2(ok2));

okWireOR # (.N(1)) wireOR (ok2, ok2x);

okWireIn  wi00(.ok1(ok1),                          .ep_addr(8'h00), .ep_dataout(value_in));
okWireOut wo20(.ok1(ok1), .ok2(ok2x[0*17 +: 17]),  .ep_addr(8'h20), .ep_datain(value_in + 16'd1));

endmodule
