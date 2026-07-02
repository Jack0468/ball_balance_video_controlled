`timescale 1ns / 1ps

module glitch_filter #(
    parameter STAGES = 2,
    parameter DEBOUNCE_CNT = 5
) (
    input wire clk,
    input wire rst,
    input wire signal_in,
    output reg signal_out
);
    // 2-stage synchronizer to prevent metastability
    reg [STAGES-1:0] sync_regs = 0;
    
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            sync_regs <= 0;
        end else begin
            sync_regs <= {sync_regs[STAGES-2:0], signal_in};
        end
    end
    
    // Debounce counter
    reg [7:0] count = 0;
    
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            signal_out <= 0;
            count <= 0;
        end else begin
            if (sync_regs[STAGES-1] == signal_out) begin
                count <= 0;
            end else begin
                if (count == DEBOUNCE_CNT - 1) begin
                    signal_out <= sync_regs[STAGES-1];
                    count <= 0;
                end else begin
                    count <= count + 1;
                end
            end
        end
    end
endmodule
