# Vitis HLS to XEM3010 (Spartan-3) Workflow Guide

Porting the `hls_hardware/` C++ modules to the Opal Kelly XEM3010 requires a **hybrid toolchain approach**. The XEM3010 uses a Xilinx Spartan-3 FPGA, which is a legacy chip. Modern tools like Vitis/Vivado **do not support** the Spartan-3 for bitstream synthesis. 

Therefore, you will use **Vitis HLS** strictly as a "C++ to Verilog" translator, and then use the legacy **Xilinx ISE 14.7** to compile that Verilog into a physical bitstream.

Here is the exact step-by-step workflow:

## Step 1: Create the Vitis HLS Project
1. Open **Vitis HLS IDE**.
2. Click **File -> New Project**.
3. Name the project `ball_balancer_hls` and choose a workspace directory.
4. **Source Files**: Click `Add Files` and select `hls_hardware/InverseKinematics.cpp` and `hls_hardware/PIDControllers.cpp`.
5. **Top Function**: You will need a single "wrapper" function that calls both the IK and PID modules. If you haven't written it yet, create a `top_level.cpp` with a function named `balancer_top`. Type `balancer_top` into the Top Function box.
6. **Testbench Files**: Skip for now (unless you have a C++ testbench).

## Step 2: The Device Selection "Hack"
Because Vitis HLS does not know what a Spartan-3 is, you must trick it:
1. In the **Part Selection** window, do *not* look for Spartan-3.
2. Select a low-end generic modern chip, such as an **Artix-7** (e.g., `xc7a35t...`) or **Spartan-7**. 
3. *Why?* We are only using Vitis to generate generic RTL (Verilog) text files. The specific logic gates of the Spartan-3 won't matter until we synthesize in ISE.
4. Finish the wizard.

## Step 3: C Synthesis (Generating Verilog)
1. In the Vitis HLS toolbar, click the **C Synthesis** button (green play icon).
2. The tool will parse your C++ code, apply the `#pragma HLS` optimizations, and compile it into hardware logic.
3. If it throws errors about `float` or dynamic memory, ensure you strictly followed the `ap_fixed` data types outlined in the Engineering Standards!

## Step 4: Export the RTL
1. Once C Synthesis completes successfully, click **Export RTL**.
2. In the dialog, select **Format: Vivado IP** or simply locate the raw files.
3. Once exported, navigate using your file explorer to your Vitis project directory: `ball_balancer_hls/solution1/syn/verilog/`.
4. **Copy all the `.v` (Verilog) files** generated in this folder.

## Step 5: Bitstream Synthesis in Xilinx ISE 14.7
1. Open **Xilinx ISE 14.7** (You must use ISE for the Spartan-3).
2. Create a new ISE Project targeting the specific chip on your XEM3010 board (usually `xc3s1500` or `xc3s400`).
3. Add the Opal Kelly `okHost` generic Verilog files provided by the Opal Kelly FrontPanel SDK.
4. **Add the `.v` files** you copied from Vitis HLS.
5. Instantiate your `balancer_top` module inside the top-level Opal Kelly wrapper, connecting the `okWireIn` outputs to your module's coordinate inputs.
6. Run **Generate Programming File** in ISE.

You now have a `.bit` file that can be loaded onto the XEM3010 using Python!
