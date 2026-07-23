`timescale 1ns / 1ps

module tb_ov7670_axi_stream();

    reg pclk = 0;
    reg vsync = 0;
    reg href = 0;
    reg [7:0] p_data = 0;
    reg config_done = 0;

    wire [15:0] m_axis_tdata;
    wire        m_axis_tvalid;
    wire        m_axis_tlast;
    wire        m_axis_tuser;
    reg         m_axis_tready = 1;

    // Instantiate the DUT
    ov7670_axi_stream uut (
        .pclk(pclk),
        .vsync(vsync),
        .href(href),
        .p_data(p_data),
        .config_done(config_done),
        .m_axis_tdata(m_axis_tdata),
        .m_axis_tvalid(m_axis_tvalid),
        .m_axis_tlast(m_axis_tlast),
        .m_axis_tuser(m_axis_tuser),
        .m_axis_tready(m_axis_tready)
    );

    // 24MHz Clock Generation (Integer delay)
    always #21 pclk = ~pclk;

    integer pixel_count = 0;

    initial begin
        $display("--- SIMULATION START ---");
        // Wait for global reset
        #1000;
        config_done = 1;
        #1000;

        // Simulate VSYNC pulse (Start of new frame)
        // Note: We MUST keep VSYNC HIGH during the active frame, otherwise
        // the camera_read FSM will trigger 'frame_done' and ignore the pixels!
        vsync = 1;
        #1000;
        
        $display("--- Starting Frame Transmission ---");

        // Simulate one horizontal line (640 pixels)
        href = 1;
        
        repeat (640) begin
            @(negedge pclk);
            p_data = 8'hAA;
            @(negedge pclk);
            p_data = 8'hBB;
            pixel_count = pixel_count + 1;
        end
        
        @(negedge pclk);
        href = 0;

        #1000;
        vsync = 0; // Trigger 'frame_done'
        #1000;

        $display("--- Frame Transmission Complete ---");
        $finish;
    end

    // Monitor AXI-Stream Outputs
    always @(posedge pclk) begin
        if (m_axis_tvalid) begin
            if (m_axis_tuser) begin
                $display("[%0t] SOF (TUSER) DETECTED! First pixel of the frame: %h", $time, m_axis_tdata);
            end
            if (m_axis_tlast) begin
                $display("[%0t] EOL (TLAST) DETECTED! Last pixel of the row (Pixel %0d). Data: %h", $time, pixel_count, m_axis_tdata);
            end
        end
    end

endmodule
