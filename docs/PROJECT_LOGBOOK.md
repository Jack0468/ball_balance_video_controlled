# Project Logbook

## 07/07/2026
### Architecture Refactor: Offloading Control to Teensy
- **Control Logic Migration**: Migrated the `DataCollectionStateMachine`, PID controller execution, and Inverse Kinematics entirely to the Teensy firmware, offloading the Python host script.
- **Fixed-Rate Control Loop**: Restructured `BallBalancingBot.ino` to run a strict 50Hz (20ms) control loop, guaranteeing stable PID `dt` timings and robust telemetry streaming.
- **Telemetry Enhancements**: Engineered a 68-byte packed binary struct for high-efficiency USB telemetry. Expanded the dataset to log 17 variables, including explicit motor targets (`theta_a`, `theta_b`, `theta_c`) and PID internal states (`integral`, `derivative`), to properly capture expert action-states for future ML Imitation Learning / Behavioral Cloning.
- **Timestamp Synchronization**: Rewrote `collect_training_data.py` to feature a daemon thread that captures the 50Hz Teensy telemetry while the main thread captures 30fps FPGA frames. Immediately timestamps incoming data on the host to create a reliable unified time-domain and perfectly sync ML labels with images.
- **Teensy Camera Removal**: Cleaned up the Teensy codebase by deleting the obsolete `CameraAcquisition` modules, as the Opal Kelly FPGA is now exclusively responsible for all video streaming.
- **Microcontroller Migration**: JMC broke the Teensy 3.6 by omitting a proper star ground on the power supply perf board, leading to a catastrophic ground loop/voltage spike. The architecture has now been migrated to an **STM32F407G-DISC1**. Firmware and hardware documentation updated to reflect STM32 compatibility (e.g., removing Teensy-specific `Serial.send_now()`).

## 03/07/2026
### FPGA Architecture & Motor Control Debugging (Opal Kelly XEM3010)
- **Directory Restructuring**: Renamed the outdated `fpga/camera_i2c` directory to `fpga/main_controller` to accurately reflect its role as the unified top-level Verilog architecture (handling camera streaming, ADC touchscreens, and 3-axis hardware PID motor control). Updated `README.md` with synthesis instructions for Xilinx ISE 14.7.
- **PLL Clock Migration**: Migrated the Verilog architecture away from the unstable USB-derived `ti_clk`. Configured the onboard `PLL22393` via Python to generate a rock-solid 48MHz `clk1` hardware clock. Implemented proper Clock Domain Crossing (CDC) synchronizers to safely pass atomic triggers and 32-bit target angles from the USB domain into the isolated `clk1` motor domain.
- **Single Motor Test Sequence**: Created `single_motor_test_top.v` and `single_motor_sequence.py` to isolate motor testing. This test perfectly replicates the legacy Arduino diagnostic sequence (commanding +90, +180, +270, +360, and 0 degree rotations) using the new hardware P-controller.
- **Solved "10-20 Degree" Stalling Bug**: Diagnosed a critical physical bug where commanded 90-degree rotations were only resulting in 10-20 degree physical movements. Discovered that the P-controller was instantly accelerating the TMC2208 to a 93kHz step rate (1750 RPM), causing the physical rotor to stall due to inertia. The motor was only catching grip during the final deceleration phase.
- **Motor Speed Clamping**: Patched `stepper_motor_controller.v` by firmly clamping the maximum P-controller speed to `700`, which strictly limits the step rate to a perfectly safe `~2 kHz`. This guarantees the physical motors never stall and successfully track all generated steps.

## 01/07/2026
### Camera Synchronization & Data Collection (OV7670 & Teensy)
- **Data Collection Robustness**: Updated `ml_vision/scripts/collect_training_data.py` to use random uniform target generation (exploring the full `-70 to 70` X and `-55 to 55` Y safe bounds) to maximize the state-space exploration for the vision model. Added serial exception handling to gracefully recover from Teensy disconnects.
- **XCLK Stabilization**: Fixed a critical clock issue in `test_ov7670.ino`. Generating a 12MHz PWM clock on the 60MHz Teensy bus created an asymmetrical 40% duty cycle, causing camera logic lockups. Changed the XCLK frequency to a perfectly symmetrical 10MHz (50% duty cycle).
- **VSYNC Pin Flexibility**: Refactored the VSYNC polling mechanism from reading hardcoded `GPIOA_PDIR` registers to using `digitalReadFast(VSYNC_PIN)`. This allows immediate software remapping of the VSYNC pin if the physical hardware pin is bent or damaged, without losing read speed.
- **Image Decoding Fix**: Discovered and patched a `uint8` integer overflow bug in `stream_viewer.py` where the numpy multiplication operations wrapped around, rendering the RGB565 images as pitch black. Switched to fast bitwise shifting `(r << 3) | (r >> 2)` which avoids overflow entirely.
- **Sync Header Desync ("Rolling Film")**: Resolved a mathematical edge-case where the camera occasionally captured a physical color that exactly matched the 4-byte serial sync header (`\xAA\xBB\xCC\xDD`), causing the Python script to prematurely slice the frame in half. Expanded the header to 8 bytes (`\xAA\xBB\xCC\xDD\xEE\xFF\x99\x88`) to mathematically eliminate false positives.
- **Framerate & Quality Enhancements**: Doubled the OV7670 internal frame rate by changing the `CLKRC` prescaler to `/1` (`0x00`). Improved raw visual clarity by activating the internal Edge Enhancement and De-noise DSP filters via the `COM16` register (`0x38`).
