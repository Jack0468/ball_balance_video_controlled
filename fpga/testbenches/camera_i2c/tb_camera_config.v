`timescale 1ns / 1ps

module tb_camera_config;

    reg clk = 0;
    always #5 clk = ~clk; // 100MHz

    reg start = 0;
    wire sioc;
    wire siod;
    wire done;

    camera_config #(
        .CLK_FREQ(100000000)
    ) uut (
        .clk(clk),
        .start(start),
        .sioc(sioc),
        .siod(siod),
        .done(done)
    );

    // Monitor internal SCCB_start to check for Bug A (start/ready race)
    wire sccb_start = uut.SCCB_start;
    wire sccb_ready = uut.SCCB_ready;

    integer tx_count = 0;

    always @(posedge clk) begin
        if (sccb_start && sccb_ready) begin
            tx_count = tx_count + 1;
        end
    end

    initial begin
        $dumpfile("tb_camera_config.vcd");
        $dumpvars(0, tb_camera_config);

        $display("Starting camera_config Bug A Race Test...");

        // Assert start
        #100;
        @(posedge clk);
        start = 1;
        @(posedge clk);
        start = 0;

        // The I2C configuration takes ~0.77s at 100kHz for 77 registers.
        // We don't want to simulate 0.77s of 100MHz clock if we don't have to,
        // but we need to see enough transactions to prove it doesn't double-fire.
        // Wait for 3 transactions to complete.
        
        while (tx_count < 3) begin
            @(posedge clk);
        end

        // Wait a little more to see if it double fires after tx 3
        #50000;

        $display("Observed %0d transactions.", tx_count);
        $display("If no double-firing occurred, Bug A is fixed.");
        $finish;
    end

endmodule
