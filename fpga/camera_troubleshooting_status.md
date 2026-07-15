# FPGA Camera Troubleshooting Status

This document summarizes the current state of our troubleshooting for the OV7670 camera integration on the Opal Kelly XEM3010 FPGA. We will use this to pick up right where we left off.

## 1. Software & Architecture Bugs (Already Fixed)

We have successfully diagnosed and committed fixes for several critical architectural bugs in the codebase:

1. **Clock Domain Crossing (CDC) Metastability**: The USB software reset was triggering the 100MHz SDRAM arbiter asynchronously. This caused the state machine to become metastable and lock up, freezing the camera after exactly one frame. **Fix**: Implemented dual-rank flip-flop synchronizers and a 10-clock cycle synchronous reset delay.
2. **PLL Clock Overclocking (48MHz vs 24MHz)**: `grab_frame.py` was erroneously configuring PLL Output 0, but the hardware UCF physically maps `P9` (camera clock) to PLL **Output 2**. The camera was receiving the raw 48MHz reference crystal frequency, surviving for ~1ms before the 200% overclock crashed the silicon. **Fix**: Rerouted the Python PLL configuration to `Output 2`.
3. **Hardware Reset Sequencing**: The camera booted up instantly with the FPGA but received no clock for several seconds until the Python script ran. This permanently glitched the SCCB I2C state machine, causing it to ignore all configuration commands (resulting in 0 VSYNCs). **Fix**: Updated `grab_frame.py` to send a clean 10ms hardware reset pulse to the camera *after* the 24MHz clock stabilizes.
4. **I2C Open-Drain Fix**: `siod` was incorrectly defined as an `output wire` in `camera_top.v`. **Fix**: Changed to `inout wire` to correctly support open-drain ACK functionality.

## 2. The Final Hardware Root Cause (Pending Physical Fix)

After applying all the software fixes, the terminal output showed:
* `PCLK` erratically jumping between 800 Hz and 32 kHz (instead of a steady 24 MHz).
* `VSYNC` pulsing exactly 1 time.
* `HREF` pulsing exactly 1 or 2 times.

**The Diagnosis:**
You mapped the camera's 3.3V power jumper wire to **`JP3-17`**. However, `JP3-17` is **FPGA IO Pin T1**, it is *not* a power rail. 
You are attempting to pull 40mA of continuous power from a single Spartan-3 logic pin (which is only rated for ~12mA digital signaling). This causes a severe voltage droop down to ~1.8V, sending the camera into a **continuous brown-out reboot loop**. Every time the camera reboots, it twitches out a single VSYNC pulse before dying again.

## 3. Next Steps (Action Items)

When we return to this task, execute the following steps in order:

1. **Physical Rewiring**: 
   * Unplug the camera's 3.3V VCC jumper wire from `JP3-17`.
   * Plug it into **`JP3-15`** or **`JP3-16`**. Both of these are true `VDD33` power rails on the Opal Kelly board that can safely supply the required current.
2. **Compile Bitstream**:
   * We have already removed the `v_sup` logic from `camera_top.v` and `xem3010_cam.ucf` since the FPGA no longer needs to artificially drive the power pin.
   * Recompile `camera_top.bit` in your Linux VM to apply this clean slate.
3. **Run Capture**:
   * Flash the bitstream and execute `grab_frame.py`. The camera will finally have true 3.3V power, a stable 24MHz clock, and a perfect reset sequence.
