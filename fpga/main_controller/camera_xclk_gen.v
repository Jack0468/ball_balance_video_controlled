`timescale 1ns / 1ps

module camera_xclk_gen(
    input wire clk_in,  // e.g. 48MHz ti_clk
    input wire rst,
    output reg clk_out  // e.g. 24MHz xclk
);
    always @(posedge clk_in or posedge rst) begin
        if (rst) begin
            clk_out <= 1'b0;
        end else begin
            clk_out <= ~clk_out;
        end
    end
endmodule
