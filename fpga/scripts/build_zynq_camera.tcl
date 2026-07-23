# build_zynq_camera.tcl
# You can source this script from anywhere in the Vivado Tcl Console!

puts "Building Zynq Camera Architecture..."

# Determine script location dynamically so it can be run from any directory
set script_dir [file dirname [file normalize [info script]]]
set proj_dir [file normalize "$script_dir/../zynq_camera_sys"]
set src_dir [file normalize "$script_dir/../camera_i2c"]

# 1. Create a clean Vivado project for the ZedBoard (this overwrites the broken one!)
create_project -force zynq_camera_sys $proj_dir -part xc7z020clg484-1
set_property board_part digilentinc.com:zedboard:part0:1.1 [current_project]
# 2. Add our hardware source files
add_files "$src_dir/ov7670_axi_stream.v"
add_files "$src_dir/camera_read.v"
add_files "$src_dir/cam_init.v"
add_files "$src_dir/cam_config.v"
add_files "$src_dir/cam_rom.v"
add_files "$src_dir/sccb_master.v"
add_files -fileset constrs_1 "$src_dir/zedboard_ov7670.xdc"

# 3. Create the Block Design
create_bd_design "camera_system"

# 4. Add the Zynq Processing System and apply ZedBoard presets
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7 processing_system7_0
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 -config {make_external "FIXED_IO, DDR" apply_board_preset "1" Master "Disable" Slave "Disable" }  [get_bd_cells processing_system7_0]

# 5. Enable the High-Performance AXI Slave Port (HP0) for Video DMA to DDR3
set_property -dict [list CONFIG.PCW_USE_S_AXI_HP0 {1}] [get_bd_cells processing_system7_0]

# 6. Enable Interrupts (so VDMA can tell ARM when a frame is done)
# We will do this manually in the GUI to avoid version-specific Tcl parameter errors
# set_property -dict [list CONFIG.PCW_USE_FABRIC_INTERRUPTS {1} CONFIG.PCW_IRQ_F2P_INTR {1}] [get_bd_cells processing_system7_0]

# 7. Add the AXI Video Direct Memory Access (VDMA) IP
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_vdma axi_vdma_0
# Configure VDMA for Write-Only (S2MM), 64-bit memory burst
set_property -dict [list CONFIG.c_m_axis_mm2s_tdata_width {16} CONFIG.c_m_axi_mm2s_data_width {32} CONFIG.c_include_mm2s {0} CONFIG.c_include_s2mm {1} CONFIG.c_s2mm_linebuffer_depth {2048} CONFIG.c_m_axi_s2mm_data_width {64}] [get_bd_cells axi_vdma_0]

# 8. Add an Interrupt Concat block
create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat xlconcat_0
set_property -dict [list CONFIG.NUM_PORTS {1}] [get_bd_cells xlconcat_0]
connect_bd_net [get_bd_pins axi_vdma_0/s2mm_introut] [get_bd_pins xlconcat_0/In0]
# We will connect the final interrupt line manually in the GUI
# connect_bd_net [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

# 9. Run Connection Automation to magically wire the AXI memory maps!
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} Master {/processing_system7_0/M_AXI_GP0} Slave {/axi_vdma_0/S_AXI_LITE} ddr_seg {Auto} intc_ip {New AXI Interconnect} master_apm {0}}  [get_bd_intf_pins axi_vdma_0/S_AXI_LITE]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} Master {/axi_vdma_0/M_AXI_S2MM} Slave {/processing_system7_0/S_AXI_HP0} ddr_seg {Auto} intc_ip {New AXI Interconnect} master_apm {0}}  [get_bd_intf_pins processing_system7_0/S_AXI_HP0]

# 10. Add our custom Camera Logic (RTL Modules)
create_bd_cell -type module -reference ov7670_axi_stream ov7670_axi_stream_0
create_bd_cell -type module -reference cam_init cam_init_0
set_property -dict [list CONFIG.CLK_F {24000000}] [get_bd_cells cam_init_0]

# 11. Create External Ports for the Camera (matching the XDC file)
create_bd_port -dir I pclk
create_bd_port -dir O xclk
create_bd_port -dir I vsync
create_bd_port -dir I href
create_bd_port -dir I -from 7 -to 0 p_data
create_bd_port -dir O sioc
create_bd_port -dir IO siod

# 12. Wire the camera inputs to our wrapper
connect_bd_net [get_bd_ports pclk] [get_bd_pins ov7670_axi_stream_0/pclk]
connect_bd_net [get_bd_ports vsync] [get_bd_pins ov7670_axi_stream_0/vsync]
connect_bd_net [get_bd_ports href] [get_bd_pins ov7670_axi_stream_0/href]
connect_bd_net [get_bd_ports p_data] [get_bd_pins ov7670_axi_stream_0/p_data]

# 13. Wire the I2C Config module
connect_bd_net [get_bd_ports sioc] [get_bd_pins cam_init_0/o_sioc]
connect_bd_net [get_bd_ports siod] [get_bd_pins cam_init_0/o_siod]
connect_bd_net [get_bd_pins cam_init_0/o_cam_init_done] [get_bd_pins ov7670_axi_stream_0/config_done]

# 14. Configure Zynq to output a 24MHz clock (FCLK_CLK1) for the Camera XCLK
set_property -dict [list CONFIG.PCW_EN_CLK1_PORT {1} CONFIG.PCW_FPGA1_PERIPHERAL_FREQMHZ {24}] [get_bd_cells processing_system7_0]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK1] [get_bd_ports xclk]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK1] [get_bd_pins cam_init_0/i_clk]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins cam_init_0/i_rstn]

# 15. Connect the AXI-Stream Video feed from our wrapper to the VDMA
connect_bd_intf_net [get_bd_intf_pins ov7670_axi_stream_0/m_axis] [get_bd_intf_pins axi_vdma_0/S_AXIS_S2MM]

# 16. Connect the VDMA stream clock to the incoming PCLK!
connect_bd_net [get_bd_ports pclk] [get_bd_pins axi_vdma_0/s_axis_s2mm_aclk]

# 17. Ensure I2C config starts automatically
create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant xlconstant_1
connect_bd_net [get_bd_pins xlconstant_1/dout] [get_bd_pins cam_init_0/i_cam_init_start]

# 18. Save Design
save_bd_design

# 19. Create HDL Wrapper
make_wrapper -files [get_files *camera_system.bd] -top
set wrapper_path "$proj_dir/zynq_camera_sys.gen/sources_1/bd/camera_system/hdl/camera_system_wrapper.v"
add_files -norecurse $wrapper_path
set_property top camera_system_wrapper [current_fileset]

puts "=========================================================="
puts "SUCCESS: ZedBoard Camera Block Design Generated Perfectly!"
puts "Next Step: Click 'Generate Bitstream' in the Flow Navigator"
puts "=========================================================="
