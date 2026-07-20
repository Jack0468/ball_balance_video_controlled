`timescale 1ns / 1ps

`default_nettype none

module top
    (   
        // from Opal Kelly
        input  wire [7:0]  hi_in,
        output wire [1:0]  hi_out,
        inout  wire [15:0] hi_inout,
        output wire        hi_muxsel,
        inout  wire        i2c_sda,
        inout  wire        i2c_scl,
	    input wire         clk1,
        

        // I/O to camera
        input wire       i_top_pclk, 
        input wire [7:0] i_top_pix_byte,
        input wire       i_top_pix_vsync,
        input wire       i_top_pix_href,

        //output wire      o_top_reset,
        //output wire      o_top_pwdn,
        output wire      o_top_xclk,
        output wire      o_top_siod,
        output wire      o_top_sioc,
        output wire      o_top_v_sup
       
    );

    //unused wires
    wire o_top_reset, o_top_pwdn;

    //target interface bus
    wire ti_clk;
    wire [30:0] ok1;
    wire [16:0] ok2;
    wire [17*8-1:0] ok2x;
    
    // from python 
    wire [15:0] i_top_cam_start; 
    wire o_top_cam_done;
    wire [15:0] i_top_rst;
    

    assign hi_muxsel = 1'b0;   // connect FPGA to the USB microcontroller, not the PROM
    assign i2c_sda   = 1'bz;   // release the I2C bus as it's irrelevant 
    assign i2c_scl   = 1'bz;

    assign o_top_v_sup = 1'b1; // power the camera	
    
    
    // Connect cam_top/vga_top modules to BRAM
    wire [11:0] i_bram_pix_data,    o_bram_pix_data;
    wire [18:0] i_bram_pix_addr,    o_bram_pix_addr; 
    wire        i_bram_pix_wr;
           
    // Reset synchronizers for all clock domains
    reg r1_rstn_top_clk,    r2_rstn_top_clk;
    reg r1_rstn_pclk,       r2_rstn_pclk;
    reg r1_rstn_clk25m,     r2_rstn_clk25m; 
        
    wire w_clk25m; 
    
    
    
    // FPGA-camera interface
    cam_top 
    #(  .CAM_CONFIG_CLK(100_000_000)    )
    OV7670_cam
    (
        .i_clk(clk1                ),
        .i_rstn_clk(r2_rstn_top_clk     ),
        .i_rstn_pclk(r2_rstn_pclk       ),
        
        // I/O for camera init
        .i_cam_start(i_top_cam_start    ),
        .o_cam_done(o_top_cam_done      ), 
        
        // I/O camera
        .i_pclk(i_top_pclk              ),
        .i_pix_byte(i_top_pix_byte      ), 
        .i_vsync(i_top_pix_vsync        ), 
        .i_href(i_top_pix_href          ),
        .o_reset(o_top_reset            ),
        .o_pwdn(o_top_pwdn              ),
        .o_siod(o_top_siod              ),
        .o_sioc(o_top_sioc              ), 
        
        // Outputs from camera to BRAM
        .o_pix_wr(                      ),
        .o_pix_data(i_bram_pix_data     ),
        .o_pix_addr(i_bram_pix_addr     )
    );
    
    mem_bram
    #(  .WIDTH(12                       ), 
        .DEPTH(640*480)                 )
     pixel_memory
     (
        // BRAM Write signals (cam_top)
        .i_wclk(i_top_pclk              ),
        .i_wr(1'b1                      ), 
        .i_wr_addr(i_bram_pix_addr      ),
        .i_bram_data(i_bram_pix_data    ),
        .i_bram_en(1'b1                 ),
         
         // BRAM Read signals (vga_top)
        .i_rclk(w_clk25m                ),
        .i_rd(1'b1                      ),
        .i_rd_addr(o_bram_pix_addr      ), 
        .o_bram_data(o_bram_pix_data    )
     );
     

    okHost okHI(
	.hi_in(hi_in), .hi_out(hi_out), .hi_inout(hi_inout), .ti_clk(ti_clk),
	.ok1(ok1), .ok2(ok2));
	
	okWireOR # (.N(7)) wireOR (ok2, ok2x);

    okWireIn  wi02(.ok1(ok1),.ep_addr(8'h02), .ep_dataout(i_top_cam_start));
	okWireIn  wi03(.ok1(ok1), .ep_addr(8'h03), .ep_dataout(i_top_rst));
    
    
endmodule
