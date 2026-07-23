setws C:/vitis
platform read C:/vitis/zedboard_platform/platform.spr
platform active zedboard_platform
platform config -updatehw C:/Users/Admin/Documents/Windows_codespace/VRI_2026/fpga/zynq_camera_sys/camera_system_wrapper.xsa
platform generate
app build -name udp_camera_app
