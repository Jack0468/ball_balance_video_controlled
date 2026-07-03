`timescale 1ns / 1ps

module single_motor_test_top (
    // Opal Kelly Host Interface
    input  wire [7:0]  hi_in,
    output wire [2:0]  hi_out,
    inout  wire [15:0] hi_inout,
    inout  wire        hi_aa,
    input  wire        clk1, // PLL0 48MHz (Robust Hardware Clock)

    // Motor 1 Pins (JP3)
    output wire        step1_pin,
    output wire        dir1_pin,
    output wire        enable_pin
);

    // Opal Kelly Host
    wire ti_clk;
    wire [30:0] ok1;
    wire [16:0] ok2;
    wire [17*3-1:0] ok2x;
    
    okHost okHI(
        .hi_in(hi_in), .hi_out(hi_out), .hi_inout(hi_inout), .hi_aa(hi_aa),
        .ti_clk(ti_clk), .ok1(ok1), .ok2(ok2)
    );
    
    okWireOR # (.N(3)) wireOR (.ok2(ok2), .ok2s(ok2x));
    
    // Always enable the TMC2208 drivers (Active-Low)
    assign enable_pin = 1'b0;

    // --- Wire In Endpoints ---
    wire [15:0] ep01_wire, ep02_wire;
    okWireIn wi01 (.ok1(ok1), .ep_addr(8'h01), .ep_dataout(ep01_wire)); // Target LSB
    okWireIn wi02 (.ok1(ok1), .ep_addr(8'h02), .ep_dataout(ep02_wire)); // Target MSB

    wire [15:0] ep40_trig;
    okTriggerIn ti40 (.ok1(ok1), .ep_addr(8'h40), .ep_clk(ti_clk), .ep_trigger(ep40_trig));
    wire update_latch = ep40_trig[0];
    wire zero_motors  = ep40_trig[1];

    // --- Clock Domain Crossing (ti_clk -> clk1) ---
    reg [2:0] update_sync;
    reg [2:0] zero_sync;
    
    always @(posedge clk1) begin
        update_sync <= {update_sync[1:0], update_latch};
        zero_sync   <= {zero_sync[1:0], zero_motors};
    end
    
    wire clk1_update = update_sync[1] & ~update_sync[2];
    wire clk1_zero   = zero_sync[1]   & ~zero_sync[2];

    // Atomic latch
    reg signed [31:0] target1 = 0;
    always @(posedge clk1) begin
        if (clk1_update) begin
            target1 <= {ep02_wire, ep01_wire};
        end
    end

    // --- Core Motor Controller ---
    wire signed [31:0] current1;
    
    stepper_motor_controller motor1(
        .clk(clk1), 
        .rst(1'b0), 
        .zero_motors(clk1_zero), 
        .target_position(target1), 
        .current_position(current1),
        .step_pin(step1_pin), 
        .dir_pin(dir1_pin)
    );

    // --- Wire Out Endpoints ---
    okWireOut wo21 (.ok1(ok1), .ok2(ok2x[ 0*17 +: 17 ]), .ep_addr(8'h21), .ep_datain(current1[15:0]));
    okWireOut wo22 (.ok1(ok1), .ok2(ok2x[ 1*17 +: 17 ]), .ep_addr(8'h22), .ep_datain(current1[31:16]));

endmodule
