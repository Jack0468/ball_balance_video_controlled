`timescale 1ns / 1ps

module OV7670_config_rom(
    input wire clk,
    input wire [7:0] addr,
    output reg [15:0] dout
    );
    //FFFF is end of rom, FFF0 is delay
    always @(posedge clk) begin
    case(addr) 

    // 0: dout <= 16'h12_80; // COM7: Reset registers
    // 1: dout <= 16'hFF_F0; // Delay marker
  
    // // Timing / Clock
    // 2: dout <= 16'h11_01; // CLKRC: Internal clock pre-scaler (0x01 = Divide by 2 -> 30fps to avoid USB bottlenecking)
    // 3: dout <= 16'h3b_0a; // COM11: Night mode, banding filter
    // 4: dout <= 16'h3a_04; // TSLB: YUYV, UYVY formatting (very important for colors)

    // // Output format (RGB565)
    // 5: dout <= 16'h12_04; // COM7: Output format RGB (Color bar disabled)
    // 6: dout <= 16'h40_10; // COM15: RGB565 output format
    // 7: dout <= 16'h8c_00; // RGB444: Disable
    // 8: dout <= 16'h3a_04; // TSLB: UYVY formatting

    // // Resolution (QQVGA)
    // 9: dout <= 16'h0c_04; // COM3: Enable scaling
    // 10: dout <= 16'h3e_1a; // COM14: Scaling PCLK and manual scaling enable
    // 11: dout <= 16'h72_22; // SCALING_DCWCTR
    // 12: dout <= 16'h73_f2; // SCALING_PCLK_DIV
    // 13: dout <= 16'h17_16; // HSTART
    // 14: dout <= 16'h18_04; // HSTOP
    // 15: dout <= 16'h32_a4; // HREF
    // 16: dout <= 16'h19_02; // VSTART
    // 17: dout <= 16'h1a_7a; // VSTOP
    // 18: dout <= 16'h03_0a; // VREF

    // // Color Matrix (crucial for RGB colors, otherwise it's just green/black)
    // 19: dout <= 16'h4f_80; // MTX1
    // 20: dout <= 16'h50_80; // MTX2
    // 21: dout <= 16'h51_00; // MTX3
    // 22: dout <= 16'h52_22; // MTX4
    // 23: dout <= 16'h53_5e; // MTX5
    // 24: dout <= 16'h54_80; // MTX6
    // 25: dout <= 16'h56_40; // MTXS
    // 26: dout <= 16'h58_9e; // MTX7
    // 27: dout <= 16'h59_88; // MTX8
    // 28: dout <= 16'h6d_55; // RSVD
    // 29: dout <= 16'h6e_11; // RSVD
    // 30: dout <= 16'h6f_9f; // RSVD
    // 31: dout <= 16'hb0_84; // RSVD

    // // Image Quality Enhancements
    // 32: dout <= 16'h41_38; // COM16: Enable Edge Enhancement, De-noise, and Color Matrix AWG

    // // Auto Exposure / Auto Gain / Auto White Balance
    // 33: dout <= 16'h13_e7; // COM8: Enable AEC, AGC, AWB
    // 34: dout <= 16'h00_00; // GAIN
    // 35: dout <= 16'h10_00; // AECH
    // 36: dout <= 16'h0d_40; // COM4
    // 37: dout <= 16'h14_18; // COM9: Automatic gain ceiling 8x
    // 38: dout <= 16'ha5_05; // RSVD
    // 39: dout <= 16'hab_07; // RSVD
    // 40: dout <= 16'h24_95; // RSVD
    // 41: dout <= 16'h25_33; // RSVD
    // 42: dout <= 16'h26_e3; // RSVD
    // 43: dout <= 16'h9f_78; // RSVD
    // 44: dout <= 16'ha0_68; // RSVD
    // 45: dout <= 16'ha1_03; // RSVD

    // 46: dout <= 16'hFF_FF; // End marker


    0:  dout <= 16'h12_80; //reset
    1:  dout <= 16'hFF_F0; //delay
    2:  dout <= 16'h12_04; // COM7,     set RGB color output
    3:  dout <= 16'h11_03; // CLKRC     divide input clock by 4 to avoid USB overflow
    4:  dout <= 16'h0C_00; // COM3,     default settings 
    5:  dout <= 16'h3E_00; // COM14,    no scaling, normal pclock 
    6:  dout <= 16'h04_00; // COM1,     disable CCIR656
    7:  dout <= 16'h40_10; //COM15,     RGB565, normal output range
    8:  dout <= 16'h3a_04; //TSLB       set correct output data sequence (magic)
    9:  dout <= 16'h0c_00; // COM3: disable scaling 
    10: dout <= 16'h3e_00; // COM14: no PCLK scaling, normal
    11: dout <= 16'h14_18; //COM9       MAX AGC value x4
    12: dout <= 16'h4F_80; //MTX1       restored from test_ov7670
    13: dout <= 16'h50_80; //MTX2
    14: dout <= 16'h51_00; //MTX3
    15: dout <= 16'h52_22; //MTX4
    16: dout <= 16'h53_5E; //MTX5
    17: dout <= 16'h54_80; //MTX6
    18: dout <= 16'h58_9E; //MTXS
    19: dout <= 16'h3D_C0; //COM13      sets gamma enable, does not preserve reserved bits, may be wrong?
    20: dout <= 16'h17_13; //HSTART     standard VGA
    21: dout <= 16'h18_01; //HSTOP      standard VGA
    22: dout <= 16'h32_B6; //HREF       standard VGA
    23: dout <= 16'h19_02; //VSTART     standard VGA
    24: dout <= 16'h1A_7A; //VSTOP      standard VGA
    25: dout <= 16'h03_0A; //VREF       standard VGA
    26: dout <= 16'h0F_41; //COM6       reset timings
    27: dout <= 16'h1E_00; //MVFP       disable mirror / flip //might have magic value of 03
    28: dout <= 16'h33_0B; //CHLF       //magic value from the internet
    29: dout <= 16'h3C_78; //COM12      no HREF when VSYNC low
    30: dout <= 16'h69_00; //GFIX       fix gain control
    31: dout <= 16'h74_00; //REG74      Digital gain control
    32: dout <= 16'hB0_84; //RSVD       magic value from the internet *required* for good color
    33: dout <= 16'hB1_0c; //ABLC1
    34: dout <= 16'hB2_0e; //RSVD       more magic internet values
    35: dout <= 16'hB3_80; //THL_ST
    //begin mystery scaling numbers
    36: dout <= 16'h70_3a;
    37: dout <= 16'h71_35;
    38: dout <= 16'h72_11;
    39: dout <= 16'h73_f0;  
    40: dout <= 16'ha2_02;
    //gamma curve values
    41: dout <= 16'h7a_20;
    42: dout <= 16'h7b_10;
    43: dout <= 16'h7c_1e;
    44: dout <= 16'h7d_35;
    45: dout <= 16'h7e_5a;
    46: dout <= 16'h7f_69;
    47: dout <= 16'h80_76;
    48: dout <= 16'h81_80;
    49: dout <= 16'h82_88;
    50: dout <= 16'h83_8f;
    51: dout <= 16'h84_96;
    52: dout <= 16'h85_a3;
    53: dout <= 16'h86_af;
    54: dout <= 16'h87_c4;
    55: dout <= 16'h88_d7;
    56: dout <= 16'h89_e8;
    //AGC and AEC
    57: dout <= 16'h13_e0; //COM8, disable AGC / AEC
    58: dout <= 16'h00_00; //set gain reg to 0 for AGC
    59: dout <= 16'h10_00; //set ARCJ reg to 0
    60: dout <= 16'h0d_40; //magic reserved bit for COM4
    61: dout <= 16'h14_18; //COM9, 4x gain + magic bit
    62: dout <= 16'ha5_05; // BD50MAX
    63: dout <= 16'hab_07; //DB60MAX
    64: dout <= 16'h24_95; //AGC upper limit
    65: dout <= 16'h25_33; //AGC lower limit
    66: dout <= 16'h26_e3; //AGC/AEC fast mode op region
    67: dout <= 16'h9f_78; //HAECC1
    68: dout <= 16'ha0_68; //HAECC2
    69: dout <= 16'ha1_03; //magic
    70: dout <= 16'ha6_d8; //HAECC3
    71: dout <= 16'ha7_d8; //HAECC4
    72: dout <= 16'ha8_f0; //HAECC5
    73: dout <= 16'ha9_90; //HAECC6
    74: dout <= 16'haa_94; //HAECC7
    75: dout <= 16'h13_e5; //COM8, enable AGC / AEC
    default: dout <= 16'hFF_FF;         //mark end of ROM
    endcase
    
    end
endmodule