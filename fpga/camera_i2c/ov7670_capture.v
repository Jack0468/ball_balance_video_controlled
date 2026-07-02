module ov7670_capture (
    input wire pclk,     // Pixel clock from the camera
    input wire vsync,    // Vertical sync from the camera (High = frame start/end)
    input wire href,     // Horizontal reference (High = active pixel data)
    input wire [7:0] d,  // 8-bit data bus from the camera
    
    output reg [14:0] addr,  // Memory address (QQVGA 160x120 = 19200, needs 15 bits)
    output reg [15:0] dout,  // 16-bit RGB565 pixel data
    output reg we            // Write enable for memory
);

    // To keep track of which byte we are receiving (High byte or Low byte of RGB565)
    reg byte_sel = 0; 
    
    always @(posedge pclk) begin
        // By default, we are not writing to memory
        we <= 1'b0;
        
        // VSYNC is active HIGH during the blanking period between frames.
        // When VSYNC is high, we reset our address and byte selector for the new frame.
        if (vsync == 1'b1) begin
            addr <= 15'd0;
            byte_sel <= 1'b0;
        end 
        else begin
            // When VSYNC is low, we are in the active frame.
            // HREF is HIGH when a valid pixel row is being transmitted.
            if (href == 1'b1) begin
                if (byte_sel == 1'b0) begin
                    // First byte (High byte of RGB565)
                    dout[15:8] <= d;
                    byte_sel <= 1'b1;
                end 
                else begin
                    // Second byte (Low byte of RGB565)
                    dout[7:0] <= d;
                    byte_sel <= 1'b0;
                    
                    // We now have a full 16-bit pixel, enable writing to memory
                    we <= 1'b1; 
                end
            end 
            else begin
                // Reset byte selector when not in a valid row (HREF is low)
                byte_sel <= 1'b0;
            end
        end
        
        // Increment the memory address immediately after we write a pixel
        if (we == 1'b1) begin
            addr <= addr + 1'b1;
        end
    end

endmodule
