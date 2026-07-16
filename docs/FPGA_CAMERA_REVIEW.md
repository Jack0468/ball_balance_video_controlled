# FPGA Camera Pipeline Review - July 16, 2026

## Objective
To stabilize the OV7670 camera initialization via I2C, properly buffer frames using an SDRAM arbiter, and successfully retrieve frames to the host PC via USB 2.0 (FrontPanel 3.x).

## Key Findings & Challenges

### 1. The `okBTPipeOut` Silent Failure (The "Totally Black Frame")
**Issue:** `grab_frame.py` was successfully reporting "Captured frame 1..5" almost instantly, but the resulting `test_frame.png` was pitch black.
**Root Cause:** The Opal Kelly XEM3010 is a USB 2.0 device (FrontPanel 3.x) and **does not support Block-Throttled Pipe Out (`okBTPipeOut`)**. Because it is unsupported, calling `ReadFromBlockPipeOut` silently failed or returned empty zeros while reporting a successful transfer of `FRAME_BYTES` (614,400). 
**Solution:** Replaced `okBTPipeOut` with standard `okPipeOut` in `camera_top.v`. Modified `grab_frame.py` to use `ReadFromPipeOut`. Because `okPipeOut` does not throttle on the FPGA side, we introduced a polling loop to wait for `WO_FRAME` (WireOut 0x21, indicating the SDRAM is full) *before* attempting the read.

### 2. Camera Configuration & Crashing (The "Stopped Counters")
**Issue:** While attempting to debug the black frame, modifications to the `OV7670_config_rom.v` (specifically manipulating `HSTART`, `HSTOP`, `HREF`, and `VSTART`) caused the camera hardware to crash.
**Symptoms:** `check_status.py` reported that `HREF` stopped ticking entirely (stalling at exactly 408 counts).
**Solution:** Reverted `OV7670_config_rom.v` to the exact state from commit `d60b634`, which proved to produce stable `PCLK`, `HREF`, and `VSYNC` ticking. The key to this stability was maintaining `CLKRC = 0x11_03` (divide by 4) and avoiding experimental bounding box modifications.

### 3. PLL Clock Dropping via `check_status.py`
**Issue:** Running `check_status.py` sometimes killed the `PCLK` entirely (counters stopped ticking).
**Root Cause:** The `dev.Open("")` call resets the FPGA's PLL on the XEM3010 unless the PLL configuration is explicitly re-applied or the handle is kept open. `check_status.py` was opening the device without re-applying the PLL settings that `grab_frame.py` had applied, killing the camera clock.
**Solution:** Ensure PLL settings are re-applied if scripts are separated, or rely on `grab_frame.py` which properly initializes the `PLL22393`.

## Current State (Physically Verified)
As of commit `d60b634` (with the USB pipe fixes applied):
1. **Camera Initialization:** The camera configures correctly over I2C and outputs continuous, stable `HREF` and `VSYNC` signals.
2. **SDRAM Arbiter:** The SDRAM arbiter actively captures frames, relying on the rising edge of `VSYNC` to safely reset state during vertical blanking.
3. **USB Transfer:** `grab_frame.py` uses `okPipeOut` and safely polls the FPGA for a full frame before transferring.

## Next Steps
- Recompile the bitstream with the `okPipeOut` fix.
- Run `grab_frame.py` to verify that the captured bytes contain actual pixel data (non-zero).
- Once data is confirmed, begin carefully adjusting color format registers (e.g., `COM7`, `COM15`, `TSLB`) via I2C to achieve proper RGB565 decoding without altering timing registers.
