# run_csynth.tcl
#
# Batch C-synthesis build for the Domain-4 PID/IK control core
# (balance_controller -> get_angles -> ik_solver x3 -> calculate_triangle).
#
# This project's standalone `vitis_hls` binary does not exist in the 2025.2
# Unified IDE install (checked: only vitis-run/v++/vitis ship under
# C:\AMDDesignTools\2025.2\Vitis\bin). HLS batch builds go through:
#
#   & "C:\AMDDesignTools\2025.2\Vitis\bin\vitis-run.bat" --mode hls --tcl run_csynth.tcl
#
# run from this directory (fpga/hls_hardware/). vitis-run's --mode hls wraps
# the classic open_project/set_top/csynth_design HLS Tcl API used below.
#
# Part is xc7z020clg484-1 (the ZedBoard part string actually used by this
# project's real Vivado builds -- see fpga/scripts/build_zynq_camera.tcl and
# fpga/vivado.log -- NOT the XC7Z020-1CSG484CES string that appears in this
# agent's own role description; that string doesn't match any real build log
# in this repo and should be treated as suspect/flagged, not reused).
#
# create_clock's period is a required HLS synthesis parameter (the PL fabric
# clock the datapath is synthesized against) -- it is NOT the same thing as
# the control-loop call-rate dt=0.001 question flagged elsewhere in
# docs/TODO.md as needing a human timing-architecture decision. 10ns/100MHz
# is used here only as a generic, commonly-used default to get a real DSP
# count; it does not resolve, and is not intended to resolve, that separate
# open decision.
#
# Output project directory is intentionally outside git (see
# fpga/hls_hardware/.gitignore) -- these are regenerable build artifacts,
# same convention as fpga/zynq_camera_sys/ being gitignored at the repo root.

set script_dir [file dirname [file normalize [info script]]]
set proj_dir   [file normalize "$script_dir/hls_proj"]

open_project -reset "$proj_dir"
set_top balance_controller
add_files "$script_dir/PIDControllers.cpp"
add_files "$script_dir/InverseKinematics.cpp"

open_solution -reset "solution1"
set_part {xc7z020clg484-1}
create_clock -period 10 -name default

csynth_design

exit
