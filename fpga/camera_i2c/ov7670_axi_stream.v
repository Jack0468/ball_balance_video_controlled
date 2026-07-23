`timescale 1ns / 1ps

module ov7670_axi_stream (
    input  wire        pclk,
    input  wire        vsync,
    input  wire        href,
    input  wire [7:0]  p_data,
    input  wire        config_done,

    // AXI4-Stream Master Interface
    output wire [15:0] m_axis_tdata,
    output wire        m_axis_tvalid,
    output wire        m_axis_tlast,
    output wire        m_axis_tuser,
    input  wire        m_axis_tready  // Ignored by camera, but required by AXI standard
);

    wire [15:0] pixel_data;
    wire        pixel_valid;
    wire        frame_start;
    wire        frame_done;

    // Instantiate the existing robust camera reader
    camera_read reader (
        .p_clock(pclk),
        .vsync(vsync),
        .href(href),
        .p_data(p_data),
        .pixel_data(pixel_data),
        .pixel_valid(pixel_valid),
        .frame_done(frame_done),
        .frame_start(frame_start),
        .config_done(config_done),
        .full(1'b0) // VDMA will absorb it, we don't backpressure the camera
    );

    // -------------------------------------------------------------------------
    // AXI-Stream TUSER (Start of Frame)
    // TUSER must be HIGH for exactly the first pixel (first TVALID) of the frame.
    // -------------------------------------------------------------------------
    reg sof = 1'b0;
    always @(posedge pclk) begin
        if (frame_start) begin
            sof <= 1'b1;
        end else if (pixel_valid) begin
            sof <= 1'b0; // clear after the very first valid pixel is emitted
        end
    end

    // -------------------------------------------------------------------------
    // AXI-Stream TLAST (End of Line)
    // VDMA expects TLAST to pulse HIGH on the last pixel of every horizontal row.
    // The camera is configured for VGA (640x480).
    // -------------------------------------------------------------------------
    reg [9:0] pixel_cnt = 0;
    always @(posedge pclk) begin
        if (frame_start) begin
            pixel_cnt <= 0;
        end else if (!href) begin
            pixel_cnt <= 0;
        end else if (pixel_valid) begin
            if (pixel_cnt < 10'd1023)
                pixel_cnt <= pixel_cnt + 1'b1;
        end
    end

    // Drop any extra jitter pixels beyond 640
    wire valid_pixel = pixel_valid && (pixel_cnt < 10'd640);

    // -------------------------------------------------------------------------
    // AXI-Stream Outputs
    // -------------------------------------------------------------------------
    assign m_axis_tdata  = pixel_data;
    assign m_axis_tvalid = valid_pixel;
    assign m_axis_tuser  = (sof & valid_pixel);
    assign m_axis_tlast  = (valid_pixel && (pixel_cnt == 10'd639)); // Exactly 640 pixels

endmodule
