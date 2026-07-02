`timescale 1ns / 1ps

module stepper_motor_controller (
    input wire clk,      // 48MHz system clock
    input wire rst,
    
    input wire [15:0] target_velocity, // Signed 16-bit velocity
    
    output reg step_pin,
    output reg dir_pin
);

    // Get absolute value for speed (magnitude)
    wire [14:0] speed = target_velocity[15] ? -target_velocity : target_velocity;
    
    // 24-bit Phase Accumulator
    // Max speed (32767) adds to accumulator. 
    // Overflow rate = (48MHz / 2^24) * 32767 = ~93.7 kHz step rate
    // Lowest speed (1) = (48MHz / 2^24) * 1 = ~2.8 Hz step rate
    reg [23:0] acc = 0;
    
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            acc <= 0;
            step_pin <= 0;
            dir_pin <= 0;
        end else begin
            dir_pin <= target_velocity[15]; // Sign bit sets direction
            
            // Only accumulate if we have a non-zero speed to prevent drifting
            if (speed > 0) begin
                acc <= acc + speed;
            end else begin
                acc <= 0;
            end
            
            // MSB of accumulator acts as a 50% duty cycle step pulse
            step_pin <= acc[23];
        end
    end

endmodule
