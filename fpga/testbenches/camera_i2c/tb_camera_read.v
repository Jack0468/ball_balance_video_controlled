`timescale 1ns / 1ps

module tb_camera_read;

    reg p_clock = 0;
    always #20 p_clock = ~p_clock; // 25MHz pixel clock

    reg vsync = 0;
    reg href = 0;
    reg [7:0] p_data = 0;
    reg config_done = 1; // tied HIGH for this test

    wire [15:0] pixel_data;
    wire pixel_valid;
    wire frame_done;
    wire frame_start;
    
    // We tie 'full' to 0 as our new SDRAM arbiter is treated as infinite for a single frame
    wire full = 0; 

    camera_read uut (
        .p_clock(p_clock),
        .vsync(vsync),
        .href(href),
        .p_data(p_data),
        .pixel_data(pixel_data),
        .pixel_valid(pixel_valid),
        .frame_done(frame_done),
        .frame_start(frame_start),
        .config_done(config_done),
        .full(full)
    );

    integer i, j;
    integer pixel_count = 0;
    integer expected_pixels = (640 * 480) - 1; // 307,199 (Row 10 drops a byte!)

    always @(posedge p_clock) begin
        if (pixel_valid) begin
            pixel_count = pixel_count + 1;
        end
    end

    initial begin
        $dumpfile("tb_camera_read.vcd");
        $dumpvars(0, tb_camera_read);

        $display("Starting camera_read simulation (Negative VSYNC polarity to match COM10=0x02)...");

        // Initial state
        vsync = 1;
        href = 0;
        p_data = 0;
        #1000;

        // VSYNC LOW for 3 lines (vertical blanking)
        vsync = 0;
        #5000; 
        
        // VSYNC HIGH (Start of active frame!)
        vsync = 1;
        #1000;

        $display("Active frame started. Simulating exactly 480 rows, 1280 p_clock pulses per row...");

        // 480 active rows
        for (i = 0; i < 480; i = i + 1) begin
            // active row
            href = 1;
            
            // INJECT STRESS TEST: Drop one byte on row 10 (simulate an odd number of bytes/lost clock)
            if (i == 10) begin
                for (j = 0; j < 1279; j = j + 1) begin
                    @(negedge p_clock); // STRESS TEST: Drive data exactly on the falling edge
                    p_data = (i + j) & 8'hFF;
                end
            end else begin
                for (j = 0; j < 1280; j = j + 1) begin
                    @(negedge p_clock); // STRESS TEST: Drive data exactly on the falling edge
                    p_data = (i + j) & 8'hFF;
                end
            end
            
            @(negedge p_clock); // Wait for the final falling edge of the row
            href = 0;
            
            // horizontal blanking
            for (j = 0; j < 144; j = j + 1) begin
                @(posedge p_clock);
            end
        end

        // Wait a bit to see frame_done
        #5000;
        
        // Assert VSYNC LOW again to enter VBLANK and trigger frame_done!
        vsync = 0;
        #5000;

        if (pixel_count == expected_pixels) begin
            $display("TEST PASSED: pixel_valid pulsed exactly %0d times.", pixel_count);
        end else begin
            $display("TEST FAILED: Expected %0d pixels, got %0d", expected_pixels, pixel_count);
        end

        $finish;
    end

endmodule
