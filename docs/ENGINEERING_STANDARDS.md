# Engineering Standards

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