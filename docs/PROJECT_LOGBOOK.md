# Project Logbook

## 01/07/2026
### Camera Synchronization & Data Collection (OV7670 & Teensy)
- **Data Collection Robustness**: Updated `ml_vision/scripts/collect_training_data.py` to use random uniform target generation (exploring the full `-70 to 70` X and `-55 to 55` Y safe bounds) to maximize the state-space exploration for the vision model. Added serial exception handling to gracefully recover from Teensy disconnects.
- **XCLK Stabilization**: Fixed a critical clock issue in `test_ov7670.ino`. Generating a 12MHz PWM clock on the 60MHz Teensy bus created an asymmetrical 40% duty cycle, causing camera logic lockups. Changed the XCLK frequency to a perfectly symmetrical 10MHz (50% duty cycle).
- **VSYNC Pin Flexibility**: Refactored the VSYNC polling mechanism from reading hardcoded `GPIOA_PDIR` registers to using `digitalReadFast(VSYNC_PIN)`. This allows immediate software remapping of the VSYNC pin if the physical hardware pin is bent or damaged, without losing read speed.
- **Image Decoding Fix**: Discovered and patched a `uint8` integer overflow bug in `stream_viewer.py` where the numpy multiplication operations wrapped around, rendering the RGB565 images as pitch black. Switched to fast bitwise shifting `(r << 3) | (r >> 2)` which avoids overflow entirely.
- **Sync Header Desync ("Rolling Film")**: Resolved a mathematical edge-case where the camera occasionally captured a physical color that exactly matched the 4-byte serial sync header (`\xAA\xBB\xCC\xDD`), causing the Python script to prematurely slice the frame in half. Expanded the header to 8 bytes (`\xAA\xBB\xCC\xDD\xEE\xFF\x99\x88`) to mathematically eliminate false positives.
- **Framerate & Quality Enhancements**: Doubled the OV7670 internal frame rate by changing the `CLKRC` prescaler to `/1` (`0x00`). Improved raw visual clarity by activating the internal Edge Enhancement and De-noise DSP filters via the `COM16` register (`0x38`).
