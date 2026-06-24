# Implementation Guide

This guide provides step-by-step instructions for developing, building, and deploying the Ball Balancing Robot.

## Phase 1: ML Pipeline Setup (Host PC)

1. **Environment Initialization**:
   Ensure your conda environment is active (`conda activate ball_balance_env`), which includes OpenCV, Ultralytics, and Pandas.
2. **Vision Model Benchmarking**:
   Run `python ml_vision/scripts/extract_frames.py` to generate sample images.
   Execute `python ml_vision/models/ml_model_benchmarks.py` to evaluate YOLO vs MobileNet SSD performance on your hardware.
3. **Audio Model Integration**:
   Once the audio team provides the pickled model/API, integrate the command listener to map spoken colors to target `(x, y)` setpoints on the platform.

## Phase 2: High-Level Synthesis (HLS)

To convert the legacy C++ code (from `ball-balancing-bot/`) into FPGA hardware:
1. Create a new AMD-Xilinx Vitis HLS project.
2. Import the C++ Inverse Kinematics and PID source files.
3. Refactor all floating-point variables in the C++ code to arbitrary precision fixed-point types (`ap_fixed`) based on the cost-benefit analysis.
4. Define the AXI-Stream or AXI-Lite interfaces for the coordinate inputs and step/dir outputs.
5. Run **C Synthesis** to generate the Verilog IP core.
6. Export the RTL as an IP package.

## Phase 3: FPGA Integration (Vivado & FrontPanel)

1. Open AMD Vivado and create a new Block Design.
2. Import the Opal Kelly `okHost` module. This module handles the USB/PCIe physical layer.
3. Import your HLS-generated IK/PID IP core.
4. Connect `okWireIn` endpoints from the `okHost` to the `x` and `y` coordinate inputs on your HLS core.
5. Route the operational and UART configuration output pins from the HLS core to the physical FPGA output pins connected to the TMC2208 stepper motor drivers.
6. Generate the Bitstream (`.bit` file).

## Phase 4: Host Communication Bridge

1. In your Python environment, install the Opal Kelly FrontPanel Python API.
2. Write the main execution loop to connect the ML models to the FPGA:
   ```python
   # Pseudocode Example
   import ok
   
   dev = ok.okCFrontPanel()
   dev.OpenBySerial("")
   dev.ConfigureFPGA("ball_balancer.bit")
   
   while True:
       target_state = audio_model.get_command()
       current_x, current_y = vision_model.get_position()
       
       dev.SetWireInValue(0x00, current_x)
       dev.SetWireInValue(0x01, current_y)
       dev.SetWireInValue(0x02, target_state)
       dev.UpdateWireIns()
   ```
