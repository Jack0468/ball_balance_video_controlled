# FPGA Camera Troubleshooting Status

This document summarizes the current state of our troubleshooting for the OV7670 camera integration on the Opal Kelly XEM3010 FPGA. We will use this to pick up right where we left off.

## 1. Software & Architecture Bugs (Fixed)

We have successfully diagnosed and committed fixes for several critical architectural bugs in the codebase:

1. **Clock Domain Crossing (CDC) Metastability**: The USB software reset was triggering the 100MHz SDRAM arbiter asynchronously. This caused the state machine to become metastable and lock up. **Fix**: Implemented dual-rank flip-flop synchronizers (`frame_rst_sync`, `rst_n_sync`) and a 10-clock cycle synchronous reset delay.
2. **PLL Clock Overclocking (48MHz vs 24MHz)**: `grab_frame.py` was erroneously configuring PLL Output 0, but the hardware UCF physically maps `P9` (camera clock) to PLL Output 2. **Fix**: Rerouted the Python PLL configuration to `Output 2`.
3. **Hardware Reset Sequencing**: The camera booted up instantly with the FPGA but received no clock for several seconds, glitching the SCCB I2C state machine. **Fix**: Updated `grab_frame.py` to send a clean 10ms hardware reset pulse *after* the 24MHz clock stabilizes.

## 2. Hardware Power Diagnosis (Resolved by User)

**The Diagnosis:** The camera's 3.3V power jumper wire was mapped to `JP3-17` (`T1`), which is an FPGA I/O pin, not a power rail, causing a brown-out reboot loop.
**Resolution:** The user has already physically rewired the camera power jumper to a true 3.3V VDD rail on the board. (Note: Previous documentation incorrectly suggested JP3-15/JP3-16 which are also I/O pins, but the user has correctly handled this).

## 3. SDRAM Buffering and Black Frame Issue (Fixed)

**The Issue**: After implementing SDRAM buffering, the camera produced a completely black frame (614,400 bytes of zeros).
**Root Cause**: The 3rd party SDRAM controller `sdram_controller.v` only asserted `busy` during READ/WRITE states (`state[4]=1`). During INIT and REFRESH sequences, `busy` was LOW. The `sdram_arbiter` assumed this meant it could issue commands, but the SDRAM controller silently dropped these commands because its FSM only processed them during `IDLE`. The arbiter incremented its pointers, dropping entire frames of pixel data during INIT or periodic chunks during REFRESH.
**Fixes Implemented**:
- **Controller Fix**: Modified `sdram_controller.v` to assert `busy <= (next != IDLE)`, correctly signalling when it cannot accept new commands.
- **Arbiter Hardening**: Rewrote `sdram_arbiter.v` to track `init_complete` and actively verify that commands are accepted by waiting for `busy` to transition HIGH then LOW.

## 4. OV7670 Image Quality Configuration (Fixed)

**The Issue**: Prior to the SDRAM bug, the camera produced a green/blue cast and dark image.
**Fixes Implemented**: Updated `OV7670_config_rom.v` with proper configuration for RGB565 mode:
- **MTX1-6**: Changed from YUV422 coefficients to true RGB565 matrix values from the Linux kernel driver.
- **COM15**: Changed output range to full `[00-FF]` (`0x40_D0`).
- **COM8/COM10**: Enabled Auto White Balance (AWB) and added PCLK HBLANK toggling suppression.

## 5. Unresolved Codebase Discrepancies (Pending Action)

Despite the fixes, there are minor discrepancies remaining:

1. **Phantom `v_sup` Logic Still Present:**
   * `xem3010_cam.ucf` still maps `NET "v_sup" LOC = "T1"`.
   * `camera_top.v` still contains `assign v_sup = 1'b0;` (This actually drives PWDN low, keeping the camera awake, so it functions correctly but the naming is confusing).

2. **I2C Open-Drain Fix is Blind:**
   * While `siod` was updated to `inout wire` in `camera_top.v`, it is passed into `camera_config.v` as an `output wire`. 
   * Functionally, the FPGA drives the line with `0` or `Z` (high-impedance), which safely avoids shorting out the camera's ACK pulse. However, because it's an output, the FPGA **never reads the ACK bit** back from the bus. 
   * **Action Required:** Decide whether to keep the I2C interface as a "fire-and-forget" blind transmission (which currently works safely) or update the submodules (`camera_config.v` and `SCCB_interface.v`) to correctly read and verify the `siod` ACK responses.

## 6. Next Steps

1. **Compile Bitstream**: Recompile `camera_top.bit` in the Linux VM (Xilinx ISE 14.7) to apply the SDRAM arbiter and I2C ROM fixes.
2. **Run Capture**: Flash the bitstream and execute `grab_frame.py`. Verify that the frames are no longer black and the colors are correct.
