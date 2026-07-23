open_project C:/Users/Admin/Documents/Windows_codespace/VRI_2026/fpga/zynq_camera_sys/zynq_camera_sys.xpr
open_bd_design {C:/Users/Admin/Documents/Windows_codespace/VRI_2026/fpga/zynq_camera_sys/zynq_camera_sys.srcs/sources_1/bd/camera_system/camera_system.bd}

# Disconnect the unbuffered pclk
disconnect_bd_net /pclk_1 [get_bd_pins ov7670_axi_stream_0/pclk]
disconnect_bd_net /pclk_1 [get_bd_pins axi_vdma_0/s_axis_s2mm_aclk]

# Add a BUFG
create_bd_cell -type ip -vlnv xilinx.com:ip:util_ds_buf pclk_bufg
set_property -dict [list CONFIG.C_BUF_TYPE {BUFG}] [get_bd_cells pclk_bufg]

# Connect pclk to BUFG input
connect_bd_net [get_bd_ports pclk] [get_bd_pins pclk_bufg/BUFG_I]

# Connect BUFG output to the IPs
connect_bd_net [get_bd_pins pclk_bufg/BUFG_O] [get_bd_pins ov7670_axi_stream_0/pclk]
connect_bd_net [get_bd_pins pclk_bufg/BUFG_O] [get_bd_pins axi_vdma_0/s_axis_s2mm_aclk]

save_bd_design
validate_bd_design

reset_run synth_1
launch_runs impl_1 -to_step write_bitstream -jobs 2
wait_on_run impl_1

write_hw_platform -fixed -include_bit -force -file C:/Users/Admin/Documents/Windows_codespace/VRI_2026/fpga/zynq_camera_sys/camera_system_wrapper.xsa
puts "BUFG added and bitstream regenerated!"
