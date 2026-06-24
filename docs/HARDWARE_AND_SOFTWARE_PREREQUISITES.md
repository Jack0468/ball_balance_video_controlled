# Hardware and Software Prerequisites

Before we transition into Phase 2 (Hardware Integration), you need to ensure the correct toolchains are installed on the Host PC and keep several physical constraints in mind when fabricating the robot.

## 1. Required Software Installations (Hybrid Toolchain)

Because you are using the **Opal Kelly XEM3010**, you are working with an older **Xilinx Spartan-3** FPGA. This significantly complicates the software toolchain, as modern tools don't support this chip, but ancient tools don't support C++ to Verilog compilation (HLS). 

You will need a **hybrid toolchain** utilizing two different Xilinx suites:

### A. Vitis HLS (from Vivado ML Edition)
- **Purpose**: You must use Vitis HLS as a standalone tool to compile our `hls_hardware/` C++ code into raw Verilog (RTL). 
- **Why?**: Xilinx ISE is officially discontinued (last release in 2013) and does not support the modern Vitis HLS compiler we need to synthesize our `ap_fixed` C++ code into Verilog.
- **Note**: Do NOT try to use the full Vivado suite to generate the bitstream, as Vivado completely dropped support for the Spartan-3 family years ago. You are only using the HLS engine to get the Verilog files.

### B. Xilinx ISE 14.7
- **Purpose**: Once you have the Verilog files from Vitis HLS, you will import them into Xilinx ISE along with the Opal Kelly FrontPanel endpoints. 
- **Why?**: ISE 14.7 is the ONLY software capable of mapping Verilog to the Spartan-3 silicon and synthesizing a `.bit` bitstream file for the XEM3010.

### C. Opal Kelly FrontPanel SDK
- **Purpose**: Provides the USB drivers and the Python API (`import ok`) used in the `fpga_bridge.py` script.
- **Python Bindings**: Copy the `ok.py` and `_ok` binaries from the SDK installation folder into your project root directory.

> [!WARNING]
> **Microcontroller vs. FPGA (The Spartan-3 Space Problem)**
> A common misconception is that an FPGA has "more compute power" or "more SRAM" than a microcontroller like the Teensy. 
> - A **Teensy (Microcontroller)** has a dedicated Floating Point Unit (FPU). You can run a complex trig function like `acos()` a million times sequentially, and it doesn't take up any extra "physical space" on the chip.
> - An **FPGA** doesn't run code sequentially; it builds physical circuits out of logic gates. Synthesizing hardware for `acos()` or `sqrt()` requires wiring together thousands of logic cells and DSP slices. If your IK calculates `acos()` for 3 arms simultaneously, the FPGA must physically construct three separate `acos()` circuits on the silicon.
> 
> The older **Spartan-3** on the XEM3010 only has ~32 DSP slices. Because we are physically limited by silicon real estate (not memory/SRAM), synthesizing trigonometric IK math directly into hardware might exceed the physical capacity of the chip. 
> 
> **Backup Plan**: If Vitis HLS generates an IP core that is too large to fit on the Spartan-3, we will move the IK/PID math out of the C++ hardware block and into the Python Host PC script. We will then purely use the FPGA to generate the raw high-frequency step pulses for the motors.



---

## 2. 3D Printing & Fabrication Notes

Since the control system relies on visual precision and fast, jerky physical movements, how you 3D print the parts will drastically affect the ML performance:

### The Balancing Platform
- **Color Contrast**: We are using Canny Edge Detection to find the four corners of the platform. The platform MUST be printed in a color that strongly contrasts with both the ball and the table beneath it (e.g., a matte black platform with a white/neon ball).
- **Surface Texture**: Do not print the platform on a heavily textured build plate. If the surface is bumpy, the ball will not roll smoothly, and the PID loop will constantly fight micro-vibrations. Use a smooth glass bed or sand the top surface after printing.

### Structural Integrity (Linkages & Mounts)
- **Material**: The arms connecting the stepper motors to the platform will experience high shearing forces when the PID loop reacts aggressively. Standard PLA is brittle and may snap during a crash. **PETG or ABS** is highly recommended for the arms.
- **Infill**: 
  - **Motor Mounts/Arms**: Print with 40-50% Gyroid or Cubic infill for maximum strength.
  - **Platform**: Print with 15-20% infill to keep it as lightweight as possible, reducing the inertia the motors have to overcome.

### Tolerances
- The joints of the parallel manipulator will likely use ball bearings. 3D printed holes almost always shrink slightly. Print a small, 5mm tall test ring before printing the entire base to ensure your bearings press-fit perfectly without cracking the plastic.

---

## 3. Hardware Components

### Motors and Drivers
- **Motors**: Nema 17 stepper motors (Model: 17HS3401S). These provide the necessary torque and speed for rapid balancing.
- **Drivers**: TMC2208 stepper motor drivers (rated for 2A). 
- **Control Interface**: The TMC2208 drivers must be wired for **UART operational control**. This allows the FPGA to configure microstepping and current dynamically via serial commands, rather than relying solely on hardwired pins or potentiometers.
