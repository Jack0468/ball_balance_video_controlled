`timescale 1ns / 1ps

module ads1675_reader(
    input wire clk,
    input wire rst,
    
    // SPI to ADC
    output reg sclk,
    output reg cs_n,
    input wire sdata,
    
    // Touchscreen Drive Pins
    output reg x_drive_en,
    output reg y_drive_en,
    
    // Output Coordinates
    output reg [31:0] touch_xy, // [31:16] = X, [15:0] = Y
    output reg valid
);

    reg [7:0] state = 0;
    reg [7:0] bit_cnt = 0;
    reg [23:0] shift_reg = 0;
    reg [15:0] x_val = 0;
    reg [15:0] y_val = 0;
    reg reading_y = 0; // 0 = reading X (driving X), 1 = reading Y (driving Y)
    
    // Clock divider for SPI (assuming 48MHz clk)
    reg [3:0] clk_div = 0;
    wire spi_tick = (clk_div == 4'd15);

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            sclk <= 0;
            cs_n <= 1;
            x_drive_en <= 0;
            y_drive_en <= 0;
            touch_xy <= 0;
            valid <= 0;
            state <= 0;
            clk_div <= 0;
            reading_y <= 0;
        end else begin
            clk_div <= clk_div + 1;
            valid <= 0;
            
            if (spi_tick) begin
                case (state)
                    0: begin
                        // Setup drives
                        cs_n <= 1;
                        sclk <= 0;
                        if (!reading_y) begin
                            x_drive_en <= 1;
                            y_drive_en <= 0;
                        end else begin
                            x_drive_en <= 0;
                            y_drive_en <= 1;
                        end
                        // Wait a bit for voltages to settle
                        if (bit_cnt < 255) begin
                            bit_cnt <= bit_cnt + 1;
                        end else begin
                            bit_cnt <= 0;
                            state <= 1;
                        end
                    end
                    
                    1: begin // Assert CS
                        cs_n <= 0;
                        bit_cnt <= 0;
                        state <= 2;
                    end
                    
                    2: begin // Clock high
                        sclk <= 1;
                        state <= 3;
                    end
                    
                    3: begin // Clock low, sample
                        sclk <= 0;
                        shift_reg <= {shift_reg[22:0], sdata};
                        if (bit_cnt < 23) begin // 24-bit ADC
                            bit_cnt <= bit_cnt + 1;
                            state <= 2;
                        end else begin
                            state <= 4;
                        end
                    end
                    
                    4: begin // Store value
                        cs_n <= 1;
                        if (!reading_y) begin
                            x_val <= shift_reg[23:8]; // Keep top 16 bits
                            reading_y <= 1;
                            state <= 0; // Go read Y
                        end else begin
                            y_val <= shift_reg[23:8];
                            touch_xy <= {x_val, shift_reg[23:8]};
                            valid <= 1;
                            reading_y <= 0;
                            state <= 0; // Go read X
                        end
                    end
                endcase
            end
        end
    end

endmodule
