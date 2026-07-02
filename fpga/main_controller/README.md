# FPGA USB Streaming & Motor Control

This directory contains the Verilog modules to synthesize the hardware bridge on the Opal Kelly XEM3010 (Spartan-3). The design handles capturing real-time video from the OV7670 camera, reading coordinates from the TI ADS1675 touchscreen ADC, and generating step pulses for the stepper motors, communicating everything to the Host PC via high-speed USB (FrontPanel API).

## Synthesis Instructions (Xilinx ISE 14.7)

Because the XEM3010 uses a legacy Xilinx Spartan-3 FPGA, you must compile this project using **Xilinx ISE 14.7** (modern Vivado releases do not support the Spartan-3 family).

### 1. Create the Project
1. Open Xilinx ISE 14.7.
2. Select **File > New Project**.
3. Name the project and set the directory to this folder (`camera_i2c_fpga`).
4. Set the family to **Spartan3** and select the specific device model matching your XEM3010 (e.g., `xc3s1500-4fg320`).

### 2. Add Source Files
1. Right-click the hierarchy window and select **Add Source...**
2. Add all `.v` files in this directory.
3. Ensure `fpga_usb_streamer.v` is set as the **Top-Level Module** (right-click it and select "Set as Top Module").

### 3. Add Opal Kelly Dependencies
You must include the proprietary FrontPanel HDL files provided by Opal Kelly:
1. Locate the `okLibrary.v` file from your Opal Kelly SDK installation and copy it into this folder. Add it to the ISE project.
2. Locate the `okCoreHarness.ngc` (or equivalent `.ngc` netlist for Spartan-3) from the SDK and copy it into this folder. Add it to the ISE project.

### 4. Constraints File (UCF)
1. You will need a User Constraints File (`.ucf`) to map the Verilog top-level ports (like `cam_d`, `step1_pin`, `adc_sclk`, etc.) to the physical pins on the XEM3010 breakout board.
2. Create or copy a `.ucf` file into the project and define the pin mappings according to your custom wiring harness.

### 5. Generate Bitstream
1. In the Processes window, double-click **Generate Programming File**.
2. Wait for synthesis, translation, mapping, and routing to complete. 
3. If successful, ISE will produce an `fpga_usb_streamer.bit` file.

## Testing the Hardware

1. Flash the generated `.bit` file onto the XEM3010 using the Opal Kelly FrontPanel GUI or via the Python API.
2. Run the `data_collection/collect_training_data.py` script on your PC. 
3. The script will establish a USB connection to the FPGA, bit-bang the I2C configuration to initialize the camera, and then begin aggressively pulling down the streaming video and ADC data over Endpoint `0xA0`.
4. You can write 16-bit velocities to Endpoints `0x01`, `0x02`, and `0x03` via Python to independently drive the three stepper motors.
