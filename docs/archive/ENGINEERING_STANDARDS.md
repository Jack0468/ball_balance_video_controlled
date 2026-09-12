# Engineering Standards

## Project Revision Strategy

The project is structured into two hardware revisions to ensure stable iterative development:
- **Baseline Revision:** Uses a **Teensy microcontroller** to handle the PID control loop and motor actuation, alongside Python for ML computation on the host PC. 
- **Future Revision (Hardware Acceleration):** Once the baseline is verified, the system will shift to an **FPGA**. The FPGA will use High-Level Synthesis (HLS) to offload the PID loop, Inverse Kinematics, and audio encoding for lower latency. The standards below for C++ and HLS apply to this future revision.

To maintain consistency across the Python software and the hardware compilation domains, all contributors must adhere to the following standards:

## 1. Python ML Standards

- **Environment Management**: All dependencies must be tracked in `environment.yml` and `requirements.txt`. Do not use manual `pip install` without updating the config files.
- **Non-blocking Execution**: The vision and audio models must operate asynchronously. Audio polling or model inference must not freeze the vision frame capture loop, otherwise, the FPGA will receive stalled coordinate updates.
- **Type Hinting**: All Python functions must include strict type hints (e.g., `def get_position() -> tuple[int, int]:`).
- **Coordinate Boundary Translation**: The Python model MUST perform all coordinate translations. Raw camera pixels must be converted into physical units (millimeters from center) within Python. The FPGA must only receive final physical coordinates, ensuring it wastes zero cycles on camera perspective transformation.

## 2. C++ to HLS Standards

Because the C++ code is destined for pure silicon synthesis on the FPGA, standard C++ software practices do not apply:
- **No Dynamic Memory**: Do not use `malloc`, `new`, or standard library containers like `std::vector` or `std::string`. All arrays must have fixed, compile-time bounds.
- **Fixed-Point Arithmetic**: Floating-point (`float`, `double`) math is strictly forbidden, as it consumes massive amounts of FPGA logic and increases PID latency. All variables must be manually refactored to arbitrary precision data types (e.g., `ap_fixed<32, 16>`) for IK and PID calculations.
- **Loop Unrolling**: Use `#pragma HLS UNROLL` or `#pragma HLS PIPELINE` on critical IK calculations to ensure they execute within the required clock latency limits.

## 3. FPGA & Verilog Standards

- **Opal Kelly Endpoints**: 
  - `0x00 - 0x1F`: Reserved for `WireIn` (Host to FPGA parameters like ML coordinates and audio targets).
  - `0x20 - 0x3F`: Reserved for `WireOut` (FPGA to Host telemetry).
- **Clock Domains**: The system must explicitly separate the FrontPanel USB clock domain (`okClk`) from the fast logic clock domain used for PWM generation. Crossing clock domains must be handled with proper synchronizers.

## 4. Data Management & File Structure (Medallion Architecture)

To ensure high-quality ML model training, all datasets (both Vision and Audio) must adhere strictly to a **Medallion Architecture** file structure:

- **`data/bronze/` (Raw)**: Raw, unvalidated data directly from the sensors. This includes messy webcam frames, continuous video captures, and raw noisy audio clips. Data here is append-only and should never be overwritten.
- **`data/silver/` (Cleansed & Processed)**: Filtered data. This includes frames where the platform has been successfully cropped/warped using the Canny preprocessor, or audio clips with noise reduction applied. It may contain auto-generated bounding box annotations that have not been human-validated.
- **`data/gold/` (ML-Ready)**: The final, curated, and human-validated datasets. Data here must be perfectly annotated and split into `train/`, `val/`, and `test/` directories, ready to be directly ingested by the YOLOv8 tracking model or the audio classification model.