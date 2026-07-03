`timescale 1ns / 1ps

module fpga_usb_streamer(
    // Opal Kelly Host Interface
    input  wire [7:0]  hi_in,
    output wire [2:0]  hi_out,
    inout  wire [15:0] hi_inout,
    inout  wire        hi_aa,
    input  wire        clk1, // PLL0 48MHz (Robust Hardware Clock)

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
    output wire        dir3_pin,
    output wire        enable_pin
);

    // Opal Kelly Host
    wire ti_clk; // 48MHz
    wire [30:0] ok1;
    wire [16:0] ok2;
    
    okHost okHI(
        .hi_in(hi_in), .hi_out(hi_out), .hi_inout(hi_inout), .hi_aa(hi_aa),
        .ti_clk(ti_clk), .ok1(ok1), .ok2(ok2)
    );
    
    // TMC2208 / A4988 Drivers require an active-low ENABLE signal.
    // By tying this to 0, we ensure the motors are always powered on and holding torque.
    assign enable_pin = 1'b0;
    
    // Wire OR for multiple endpoints 
    // 0: I2C SDA
    // 1: Motor 1 Pos LSB
    // 2: Motor 1 Pos MSB
    // 3: Motor 2 Pos LSB
    // 4: Motor 2 Pos MSB
    // 5: Motor 3 Pos LSB
    // 6: Motor 3 Pos MSB
    // 7: PipeOut (Video)
    wire [17*8-1:0] ok2x;
    okWireOR # (.N(8)) wireOR (.ok2(ok2), .ok2s(ok2x));
    
    // --- 1. Camera Clock Generation (24MHz) ---
    wire reset = 1'b0; // System reset can be added if needed
    wire sys_rst = reset;
    
    camera_xclk_gen xclk_gen(
        .clk_in(clk1), // Use the robust PLL clock!
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
    wire [15:0] ep01_wire, ep02_wire;
    wire [15:0] ep03_wire, ep04_wire;
    wire [15:0] ep05_wire, ep06_wire;
    
    okWireIn wi01 (.ok1(ok1), .ep_addr(8'h01), .ep_dataout(ep01_wire)); // Motor 1 Target LSB
    okWireIn wi02 (.ok1(ok1), .ep_addr(8'h02), .ep_dataout(ep02_wire)); // Motor 1 Target MSB
    okWireIn wi03 (.ok1(ok1), .ep_addr(8'h03), .ep_dataout(ep03_wire)); // Motor 2 Target LSB
    okWireIn wi04 (.ok1(ok1), .ep_addr(8'h04), .ep_dataout(ep04_wire)); // Motor 2 Target MSB
    okWireIn wi05 (.ok1(ok1), .ep_addr(8'h05), .ep_dataout(ep05_wire)); // Motor 3 Target LSB
    okWireIn wi06 (.ok1(ok1), .ep_addr(8'h06), .ep_dataout(ep06_wire)); // Motor 3 Target MSB

    wire [15:0] ep40_trig;
    okTriggerIn ti40 (.ok1(ok1), .ep_addr(8'h40), .ep_clk(ti_clk), .ep_trigger(ep40_trig));
    wire update_latch = ep40_trig[0];
    wire zero_motors  = ep40_trig[1];

    reg signed [31:0] target1 = 0;
    reg signed [31:0] target2 = 0;
    reg signed [31:0] target3 = 0;

    // Clock Domain Crossing (CDC) for triggers: ti_clk -> clk1
    reg [2:0] update_sync;
    reg [2:0] zero_sync;
    
    always @(posedge clk1) begin
        update_sync <= {update_sync[1:0], update_latch};
        zero_sync   <= {zero_sync[1:0], zero_motors};
    end
    
    wire clk1_update = update_sync[1] & ~update_sync[2];
    wire clk1_zero   = zero_sync[1]   & ~zero_sync[2];

    // Atomic update of target angles in the clk1 domain
    always @(posedge clk1) begin
        if (clk1_update) begin
            target1 <= {ep02_wire, ep01_wire};
            target2 <= {ep04_wire, ep03_wire};
            target3 <= {ep06_wire, ep05_wire};
        end
    end

    wire signed [31:0] current1, current2, current3;

    stepper_motor_controller motor1(
        .clk(clk1), 
        .rst(sys_rst), 
        .zero_motors(clk1_zero), 
        .target_position(target1), 
        .current_position(current1),
        .step_pin(step1_pin), 
        .dir_pin(dir1_pin)
    );
    
    stepper_motor_controller motor2(
        .clk(clk1), 
        .rst(sys_rst), 
        .zero_motors(clk1_zero), 
        .target_position(target2), 
        .current_position(current2),
        .step_pin(step2_pin), 
        .dir_pin(dir2_pin)
    );
    
    stepper_motor_controller motor3(
        .clk(clk1), 
        .rst(sys_rst), 
        .zero_motors(clk1_zero), 
        .target_position(target3), 
        .current_position(current3),
        .step_pin(step3_pin), 
        .dir_pin(dir3_pin)
    );

    // WireOuts to read exact current position of each motor
    okWireOut wo21 (.ok1(ok1), .ok2(ok2x[ 1*17 +: 17 ]), .ep_addr(8'h21), .ep_datain(current1[15:0]));
    okWireOut wo22 (.ok1(ok1), .ok2(ok2x[ 2*17 +: 17 ]), .ep_addr(8'h22), .ep_datain(current1[31:16]));
    okWireOut wo23 (.ok1(ok1), .ok2(ok2x[ 3*17 +: 17 ]), .ep_addr(8'h23), .ep_datain(current2[15:0]));
    okWireOut wo24 (.ok1(ok1), .ok2(ok2x[ 4*17 +: 17 ]), .ep_addr(8'h24), .ep_datain(current2[31:16]));
    okWireOut wo25 (.ok1(ok1), .ok2(ok2x[ 5*17 +: 17 ]), .ep_addr(8'h25), .ep_datain(current3[15:0]));
    okWireOut wo26 (.ok1(ok1), .ok2(ok2x[ 6*17 +: 17 ]), .ep_addr(8'h26), .ep_datain(current3[31:16]));

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
    wire pipe_read_actual;
    
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
    
    assign pipe_read_actual = pipe_read && (read_counter >= 4);
    
    assign pipe_datain = (read_counter == 0) ? 16'hAABB :
                         (read_counter == 1) ? 16'hCCDD :
                         (read_counter == 2) ? touch_xy[31:16] : // Touch X
                         (read_counter == 3) ? touch_xy[15:0]  : // Touch Y
                         fifo_dout;                              // Image Data
    
    // PipeOut Endpoint (0xA0)
    okPipeOut poA0 (.ok1(ok1), .ok2(ok2x[ 7*17 +: 17 ]), .ep_addr(8'hA0), .ep_read(pipe_read), .ep_datain(pipe_datain));
    
endmodule
