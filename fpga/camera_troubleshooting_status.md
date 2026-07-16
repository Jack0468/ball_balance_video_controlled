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

## 3. Unresolved Codebase Discrepancies (Pending Action)

Despite the physical fix and previous notes, there are two major discrepancies remaining in the codebase that the next agent must address:

1. **Phantom `v_sup` Logic Still Present:**
   * Previous documentation claimed the `v_sup` logic was removed. **This is false.**
   * `xem3010_cam.ucf` still maps `NET "v_sup" LOC = "T1"`.
   * `camera_top.v` still contains `assign v_sup = 1'b0;`.
   * **Action Required:** Remove this logic entirely, as the camera is now powered externally and leaving it in actively pulls `T1` to ground.

2. **I2C Open-Drain Fix is Blind:**
   * While `siod` was updated to `inout wire` in `camera_top.v`, it is passed into `camera_config.v` as an `output wire`. 
   * Functionally, the FPGA drives the line with `0` or `Z` (high-impedance), which safely avoids shorting out the camera's ACK pulse. However, because it's an output, the FPGA **never reads the ACK bit** back from the bus. 
   * **Action Required:** Decide whether to keep the I2C interface as a "fire-and-forget" blind transmission (which currently works safely) or update the submodules (`camera_config.v` and `SCCB_interface.v`) to correctly read and verify the `siod` ACK responses.

## 4. Next Steps

1. **Code Cleanup**: Remove the leftover `v_sup` assignments in `camera_top.v` and `xem3010_cam.ucf`.
2. **Review I2C Architecture**: Determine if the blind SCCB transmission is acceptable or if true bidirectional ACK verification is needed.
3. **Compile Bitstream**: Recompile `camera_top.bit` in the Linux VM to apply the clean slate.
4. **Run Capture**: Flash the bitstream and execute `grab_frame.py`.

## 5. FPGA Camera Stability Verification (Latest Learnings)

The following fixes and architectural learnings were successfully applied to the FPGA workspace in the last session to produce stable frames:

* **USB 2.0 `okBTPipeOut` Incompatibility**: Diagnosed that `ReadFromBlockPipeOut` is fundamentally incompatible with the Cypress FX2 hardware on the XEM3010 FPGA. Attempting to use a block-throttled pipe (`okBTPipeOut`) caused `grab_frame.py` to immediately return 614,400 bytes of empty zeros, resulting in a completely black image despite the camera capturing properly and the FPGA counters (PCLK, HREF, VSYNC) continuously ticking. 
* **Unthrottled SDRAM Buffering (The Fix)**: Resolved the silent failure by switching back to the standard unthrottled `okPipeOut`. To handle potential USB underruns without throttling, the entire 640x480 frame is buffered in the 100MHz SDRAM arbiter. The Python host script now polls the FPGA state machine via `WO_FRAME` and only initiates the unthrottled USB pipe read once a full frame is guaranteed to be sitting in SDRAM. This successfully produced continuous, uncorrupted frame captures.
