`timescale 1ns / 1ps

module fpga_usb_streamer(
    // Opal Kelly Host Interface
    input  wire [7:0]  hi_in,
    output wire [2:0]  hi_out,
    inout  wire [15:0] hi_inout,
    inout  wire        hi_aa,

    // Camera Signals
    output wire        cam_xclk,
    input  wire        pclk,
    input  wire        vsync,
    input  wire        href,
    input  wire [7:0]  cam_d,
    
    // I2C to Camera (Bit-banged from Python)
    output wire        cam_scl,
    inout  wire        cam_sda,
    output wire        cam_reset,
    
    // Touchscreen ADC Signals (TI ADS1675)
    output wire        adc_sclk,
    output wire        adc_cs_n,
    input  wire        adc_sdata,
    output wire        x_drive,
    output wire        y_drive,
    
    // Stepper Motor Control Pins (3 Motors for Delta/Platform)
    output wire        step1_pin,
    output wire        dir1_pin,
    output wire        step2_pin,
    output wire        dir2_pin,
    output wire        step3_pin,
    output wire        dir3_pin
);

    // Opal Kelly Host
    wire ti_clk; // 48MHz
    wire [30:0] ok1;
    wire [16:0] ok2;
    
    okHost okHI(
        .hi_in(hi_in), .hi_out(hi_out), .hi_inout(hi_inout), .hi_aa(hi_aa),
        .ti_clk(ti_clk), .ok1(ok1), .ok2(ok2)
    );
    
    // Wire OR for multiple endpoints (1x PipeOut, 1x WireOut = 2 endpoints)
    wire [17*2-1:0] ok2x;
    okWireOR # (.N(2)) wireOR (.ok2(ok2), .ok2s(ok2x));
    
    // --- 1. Camera Clock Generation (24MHz) ---
    wire reset = 1'b0; // System reset can be added if needed
    
    camera_xclk_gen xclk_gen(
        .clk_in(ti_clk),
        .rst(reset),
        .clk_out(cam_xclk)
    );

    // --- 2. Signal Debouncing (Glitch Filters) ---
    wire vsync_clean;
    wire href_clean;
    
    glitch_filter #(.STAGES(2), .DEBOUNCE_CNT(5)) filter_vsync(
        .clk(pclk),
        .rst(reset),
        .signal_in(vsync),
        .signal_out(vsync_clean)
    );
    
    glitch_filter #(.STAGES(2), .DEBOUNCE_CNT(5)) filter_href(
        .clk(pclk),
        .rst(reset),
        .signal_in(href),
        .signal_out(href_clean)
    );

    // --- 3. Stepper Motor Control ---
    wire [15:0] ep01_wire;
    wire [15:0] ep02_wire;
    wire [15:0] ep03_wire;
    okWireIn wi01 (.ok1(ok1), .ep_addr(8'h01), .ep_dataout(ep01_wire)); // Motor 1 Velocity
    okWireIn wi02 (.ok1(ok1), .ep_addr(8'h02), .ep_dataout(ep02_wire)); // Motor 2 Velocity
    okWireIn wi03 (.ok1(ok1), .ep_addr(8'h03), .ep_dataout(ep03_wire)); // Motor 3 Velocity

    stepper_motor_controller motor1(
        .clk(ti_clk), .rst(reset), .target_velocity(ep01_wire), 
        .step_pin(step1_pin), .dir_pin(dir1_pin)
    );
    
    stepper_motor_controller motor2(
        .clk(ti_clk), .rst(reset), .target_velocity(ep02_wire), 
        .step_pin(step2_pin), .dir_pin(dir2_pin)
    );
    
    stepper_motor_controller motor3(
        .clk(ti_clk), .rst(reset), .target_velocity(ep03_wire), 
        .step_pin(step3_pin), .dir_pin(dir3_pin)
    );

    // --- 4. I2C Bit-banging via WireIn ---
    wire [15:0] ep00_wire;
    okWireIn wi00 (.ok1(ok1), .ep_addr(8'h00), .ep_dataout(ep00_wire));
    
    assign cam_scl = ep00_wire[0];
    wire sda_out   = ep00_wire[1];
    wire sda_oe    = ep00_wire[2];
    assign cam_reset = ep00_wire[3];
    
    assign cam_sda = sda_oe ? sda_out : 1'bz;
    
    okWireOut wo20 (.ok1(ok1), .ok2(ok2x[ 0*17 +: 17 ]), .ep_addr(8'h20), .ep_datain({15'd0, cam_sda}));

    // --- 5. ADC Touchscreen Reader ---
    wire [31:0] touch_xy;
    wire touch_valid;
    
    ads1675_reader adc_reader(
        .clk(ti_clk),
        .rst(ep00_wire[4]), // Reset via bit 4 of ep00
        .sclk(adc_sclk),
        .cs_n(adc_cs_n),
        .sdata(adc_sdata),
        .x_drive_en(x_drive),
        .y_drive_en(y_drive),
        .touch_xy(touch_xy),
        .valid(touch_valid)
    );

    // --- 6. Camera Capture ---
    wire [14:0] pixel_addr;
    wire [15:0] pixel_data;
    wire pixel_we;
    
    ov7670_capture cam_cap(
        .pclk(pclk),
        .vsync(vsync_clean), // Use clean vsync
        .href(href_clean),   // Use clean href
        .d(cam_d),
        .addr(pixel_addr),
        .dout(pixel_data),
        .we(pixel_we)
    );
    
    // --- 7. FIFO for Clock Domain Crossing ---
    wire [15:0] fifo_dout;
    wire fifo_empty;
    wire fifo_full;
    wire pipe_read;
    
    // Drop frame if FIFO overflows
    reg fifo_rst = 1;
    always @(posedge pclk) begin
        if (vsync_clean) begin
            fifo_rst <= 1; // Hold reset during vertical sync to clear old data
        end else if (fifo_full) begin
            fifo_rst <= 1; // Drop frame on overflow
        end else begin
            fifo_rst <= 0;
        end
    end
    
    camera_fifo fifo(
        .wr_clk(pclk),
        .wr_rst(fifo_rst),
        .wr_en(pixel_we),
        .wr_data(pixel_data),
        .full(fifo_full),
        
        .rd_clk(ti_clk),
        .rd_rst(fifo_rst),
        .rd_en(pipe_read_actual),
        .rd_data(fifo_dout),
        .empty(fifo_empty)
    );
    
    // --- 8. Output Stream Multiplexer ---
    reg [15:0] read_counter = 0;
    wire [15:0] pipe_datain;
    
    always @(posedge ti_clk) begin
        if (vsync_clean) begin
            read_counter <= 0;
        end else if (pipe_read) begin
            read_counter <= read_counter + 1;
        end
    end
    
    wire pipe_read_actual = pipe_read && (read_counter >= 4);
    
    assign pipe_datain = (read_counter == 0) ? 16'hAABB :
                         (read_counter == 1) ? 16'hCCDD :
                         (read_counter == 2) ? touch_xy[31:16] : // Touch X
                         (read_counter == 3) ? touch_xy[15:0]  : // Touch Y
                         fifo_dout;                              // Image Data
    
    // PipeOut Endpoint (0xA0)
    okPipeOut poA0 (.ok1(ok1), .ok2(ok2x[ 1*17 +: 17 ]), .ep_addr(8'hA0), .ep_read(pipe_read), .ep_datain(pipe_datain));
    
endmodule
