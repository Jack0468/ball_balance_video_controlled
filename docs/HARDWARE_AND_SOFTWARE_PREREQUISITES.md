# Hardware and Software Prerequisites

Before we transition into Phase 2 (Hardware Integration), you need to ensure the correct toolchains are installed on the Host PC and keep several physical constraints in mind when fabricating the robot.

## 1. Required Software Installations (Unified Toolchain)

We are using the **ZedBoard (Xilinx Zynq-7000 EPP)** for this project. Unlike legacy Spartan-3 boards, the Zynq-7000 is fully supported by modern toolchains, allowing us to use a single, unified environment for all tasks.

You will need the following installed:

### A. Xilinx Vitis / Vivado 2025.2 (ML Edition)
- **Purpose**: We use the modern Vitis 2025.2 suite for the entire FPGA workflow:
  1. **High-Level Synthesis (HLS)**: Compiling our `hls_hardware/` C++ code into raw Verilog (RTL).
  2. **Simulation (`xsim`)**: All testbenches are simulated natively using Vivado Simulator.
  3. **Synthesis & Implementation**: Vivado handles the entire synthesis, placement, routing, and bitstream (`.bit`) generation natively for the Zynq XC7Z020 silicon.

> [!NOTE]
> **Microcontroller vs. FPGA (The Zynq Advantage)**
> A common constraint on older FPGAs (like the Spartan-3) was a lack of physical DSP slices to implement complex math (like trigonometry for Inverse Kinematics). 
> 
> The **Zynq-7000 (XC7Z020)** completely eliminates this problem. Not only does it have 220 dedicated DSP slices (plenty for hardware math), but it also contains a hardened **Dual-Core ARM Cortex-A9 Processing System (PS)**. If an IK algorithm is too complex for raw logic gates, we can effortlessly run it in C++ on the embedded ARM core and pass the results to the programmable logic (PL) over the high-speed AXI bus.

---

## 2. 3D Printing & Fabrication Notes

Since the control system relies on visual precision and fast, jerky physical movements, how you 3D print the parts will drastically affect the ML performance:

### The Balancing Platform
- **Color Contrast**: We are using Canny Edge Detection to find the four corners of the platform. The platform MUST be printed in a color that strongly contrasts with both the ball and the table beneath it (e.g., a matte black platform with a white/neon ball).
- **Surface Texture**: Do not print the platform on a heavily textured build plate. If the surface is bumpy, the ball will not roll smoothly, and the PID loop will constantly fight micro-vibrations. Use a smooth glass bed or sand the top surface after printing.

### Structural Integrity (Linkages & Mounts)
- **Material**: The arms connecting the stepper motors to the platform will experience high shearing forces when the PID loop reacts aggressively. Standard PLA is brittle and may snap during a crash. **PETG or ABS** is highly recommended for the arms.
- Note that the original design uses PLA.
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
- **Control Interface**: The TMC2208 drivers will be controlled directly via their **STEP and DIR pins**. UART configuration is too slow for real-time motion and will not be used. Microstepping and current limits should be configured via the hardware pins and the onboard potentiometers.

### Camera
OV7670 
