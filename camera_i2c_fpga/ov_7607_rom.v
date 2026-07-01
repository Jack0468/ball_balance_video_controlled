module ov7607_setup_rom(
	
	input clk,
	input [7:0] address,
	output [15:0] data
	);

	always @(posedge clk) begin
		case(address)
			// System Reset
			6'd0:  data <= 16'h1280; // COM7: Reset registers
			6'd1:  data <= 16'hFFFF; // Delay marker
			
			// Timing / Clock
			6'd2:  data <= 16'h1101; // CLKRC: Internal clock pre-scaler
			6'd3:  data <= 16'h3b0a; // COM11: Night mode, banding filter
			6'd4:  data <= 16'h3a04; // TSLB: YUYV, UYVY formatting
			
			// Output format (RGB565)
			6'd5:  data <= 16'h1204; // COM7: Output format RGB (Color bar disabled)
			6'd6:  data <= 16'h4010; // COM15: RGB565 output format
			6'd7:  data <= 16'h8c00; // RGB444: Disable
			6'd8:  data <= 16'h3a04; // TSLB: UYVY formatting
			
			// Resolution (QQVGA)
			6'd9:  data <= 16'h0c04; // COM3: Enable scaling
			6'd10: data <= 16'h3e1a; // COM14: Scaling PCLK and manual scaling enable
			6'd11: data <= 16'h7222; // SCALING_DCWCTR
			6'd12: data <= 16'h73f2; // SCALING_PCLK_DIV
			6'd13: data <= 16'h1716; // HSTART
			6'd14: data <= 16'h1804; // HSTOP
			6'd15: data <= 16'h32a4; // HREF
			6'd16: data <= 16'h1902; // VSTART
			6'd17: data <= 16'h1a7a; // VSTOP
			6'd18: data <= 16'h030a; // VREF
			
			// Color Matrix
			6'd19: data <= 16'h4f80;
			6'd20: data <= 16'h5080;
			6'd21: data <= 16'h5100;
			6'd22: data <= 16'h5222;
			6'd23: data <= 16'h535e;
			6'd24: data <= 16'h5480;
			6'd25: data <= 16'h5640;
			6'd26: data <= 16'h589e;
			6'd27: data <= 16'h5988;
			6'd28: data <= 16'h5a88;
			6'd29: data <= 16'h5b44;
			6'd30: data <= 16'h5c67;
			6'd31: data <= 16'h5d49;
			6'd32: data <= 16'h5e0e;
			6'd33: data <= 16'h6900;
			6'd34: data <= 16'h6a40;
			6'd35: data <= 16'h6b0a;
			6'd36: data <= 16'h6c0a;
			6'd37: data <= 16'h6d55;
			6'd38: data <= 16'h6e11;
			6'd39: data <= 16'h6f9f;
			6'd40: data <= 16'hb084;
			
			// Image Quality Enhancements
			6'd41: data <= 16'h4138; // COM16: Edge Enhancement, De-noise, AWG

			// Auto Exposure / Auto Gain / Auto White Balance
			6'd42: data <= 16'h13e7; // COM8: Enable AEC, AGC, AWB
			6'd43: data <= 16'h0000; // GAIN
			6'd44: data <= 16'h1000; // AECH
			6'd45: data <= 16'h0d40; // COM4
			6'd46: data <= 16'h1418; // COM9: Auto gain ceiling 8x
			6'd47: data <= 16'ha505;
			6'd48: data <= 16'hab07;
			6'd49: data <= 16'h2495;
			6'd50: data <= 16'h2533;
			6'd51: data <= 16'h26e3;
			6'd52: data <= 16'h9f78;
			6'd53: data <= 16'ha068;
			6'd54: data <= 16'ha103;
			
			// End marker
			6'd55: data <= 16'hFFFF; 

			// Default fallback to prevent latch generation
			default: data <= 16'hFFFF; 
	  endcase
	end
endmodule
				