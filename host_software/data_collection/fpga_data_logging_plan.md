# FPGA Data Logging Implementation Plan

This document contains the implementation plan and necessary context for adding motor command logging to the ML data collection pipeline. We are logging the exact motor actions (Pitch, Roll, and Motor Thetas) that correspond to each camera frame and ball position.

## Context and Architecture Notes
- **Opal Kelly FrontPanel API**: The communication between the PC and the FPGA happens via the `FPGABridge` class in `fpga_bridge.py`. It uses `SetWireInValue` to send data and will need to use `UpdateWireOuts` and `GetWireOutValue` to receive data back from the FPGA.
- **Fixed-Point Math**: The HLS C++ code uses `ap_fixed` (represented as `fixed_t`). In the Python bridge, values are converted from floats to Q16.16 fixed-point integers (multiply by `2^16`). When reading the output angles from the FPGA, the integer values must be converted back to floats (divide by `2^16`).
- **Teensy vs. FPGA**: Although the Teensy camera code generates the initial coordinate data, we are exclusively modifying the FPGA codebase (`hls_hardware` and `fpga_bridge.py`) for the motor logging to match the final hardware architecture.

---

## Proposed Changes

### 1. HLS Hardware Definition (`hls_hardware/`)

#### Modify `PIDControllers.h`
- Add two new reference parameters to the `balance_controller` signature: `fixed_t &out_pitch` and `fixed_t &out_roll`.

#### Modify `PIDControllers.cpp`
- Add the corresponding `#pragma HLS INTERFACE s_axilite` directives for the two new ports so they compile into AXI-Lite registers (which map to Opal Kelly WireOuts in the top-level Verilog).
- Assign the calculated `angle_y` to `out_pitch` and `angle_x` to `out_roll` right before the Inverse Kinematics calculation.

### 2. Python FPGA Bridge (`fpga_bridge.py`)

- **Endpoints**: Define new `WireOut` endpoint addresses (e.g., `0x20` to `0x24`) for the 5 outputs.
- **Conversion Helper**: Add a helper function `fixed_wire_to_float(value)` to reverse the Q16.16 fixed-point conversion.
- **Read Method**: Add a method `read_motor_commands()` that:
  1. Calls `self.dev.UpdateWireOuts()`.
  2. Fetches the 5 values via `GetWireOutValue()`.
  3. Converts them to floats.
  4. Returns them as a tuple `(pitch, roll, thetaA, thetaB, thetaC)`.

### 3. ML Data Collection Script (`ml_vision/scripts/collect_training_data.py`)

- Import and initialize `FPGABridge`.
- Inside the main data collection loop, after unpacking the `touch_x` and `touch_y` from the Teensy serial stream, pass them to `bridge.send_coordinates()`.
- Immediately call `bridge.read_motor_commands()` to retrieve the Pitch, Roll, and Motor A/B/C angles.
- Update the CSV writer logic to include three new columns (`Pitch`, `Roll`, `ThetaA`, `ThetaB`, `ThetaC`) and append these 5 new values to the row data.

## Verification Steps
1. Re-run Vitis HLS synthesis on `hls_hardware` to ensure the new ports compile correctly into Verilog.
2. Run `python collect_training_data.py` (which will fallback to `MockFrontPanel` if the physical FPGA isn't plugged in) and verify that `labels.csv` is populated with the new columns containing valid data.
