open_project C:/Users/Admin/Documents/Windows_codespace/VRI_2026/fpga/zynq_camera_sys/zynq_camera_sys.xpr
reset_run synth_1
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1
write_hw_platform -fixed -include_bit -force -file C:/Users/Admin/Documents/Windows_codespace/VRI_2026/fpga/zynq_camera_sys/camera_system_wrapper.xsa
puts "Bitstream regeneration complete!"
