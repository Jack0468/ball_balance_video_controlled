`timescale 1ns / 1ps

module stepper_motor_controller (
    input wire clk,      // 48MHz system clock
    input wire rst,
    input wire zero_motors,
    
    input wire signed [31:0] target_position,
    
    output reg step_pin,
    output reg dir_pin,
    output reg signed [31:0] current_position
);

    wire signed [31:0] error = target_position - current_position;
    wire [31:0] abs_error = (error < 0) ? -error : error;
    
    // Proportional speed control (P-controller)
    // Multiply error by 64 (shift left by 6).
    wire [31:0] p_term = abs_error << 6;
    
    // Clamp max speed to 32767 (15-bit max for our accumulator)
    wire [14:0] speed = (p_term > 32767) ? 15'd32767 : p_term[14:0];
    
    reg [23:0] acc = 0;
    reg last_step_pin = 0;
    
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            acc <= 0;
            step_pin <= 0;
            dir_pin <= 0;
            current_position <= 0;
            last_step_pin <= 0;
        end else if (zero_motors) begin
            current_position <= 0;
            acc <= 0;
            step_pin <= 0;
            last_step_pin <= 0;
        end else begin
            last_step_pin <= step_pin;
            
            if (error > 0) begin
                dir_pin <= 1'b1; // Positive direction
            end else if (error < 0) begin
                dir_pin <= 1'b0; // Negative direction
            end
            
            // Only accumulate if we have not reached target
            if (abs_error > 0) begin
                acc <= acc + speed;
            end else begin
                acc <= 0;
            end
            
            step_pin <= acc[23];
            
            // On rising edge of step_pin, update current position
            if (step_pin && !last_step_pin) begin
                if (dir_pin == 1'b1)
                    current_position <= current_position + 1;
                else
                    current_position <= current_position - 1;
            end
        end
    end

endmodule
