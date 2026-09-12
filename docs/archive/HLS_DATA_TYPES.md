# Cost-Benefit Analysis: HLS Data Types (Float vs. Fixed-Point)

When synthesizing C++ into Verilog using Xilinx Vitis HLS, the choice of data types for the Inverse Kinematics (IK) and PID calculations is the most critical design decision for FPGA performance.

## Option A: Standard Floating Point (`float` / `double`)

**Benefits:**
- **Zero Friction**: You can copy/paste the existing Teensy C++ code directly into HLS without changing any variable types.
- **No Overflow Risk**: The massive dynamic range of floats guarantees your PID accumulators and IK angles will never overflow or lose precision.
- **Faster Prototyping**: You skip the complex mathematical analysis phase entirely.

**Costs:**
- **Resource Hog**: Floating-point math on an FPGA does not happen natively (unless the specific FPGA has hardened float DSPs). It requires synthesizing complex IEEE-754 state machines. This consumes a massive percentage of your logic gates (LUTs/FFs).
- **High Latency**: Floating-point operations take many clock cycles to complete, significantly increasing the latency of the PID feedback loop.

---

## Option B: Arbitrary Precision Fixed-Point (`ap_fixed<W, I>`)

*Where `W` is the total number of bits, and `I` is the number of integer bits (e.g., `ap_fixed<32, 16>` gives 16 bits of whole numbers and 16 bits of decimals).*

**Benefits:**
- **Blazing Fast**: Fixed-point math is essentially just integer math to the FPGA. It executes in a single clock cycle.
- **Highly Efficient**: Consumes very few logic cells and maps perfectly to the FPGA's built-in DSP slices. Leaves plenty of room on the FPGA for other tasks.

**Costs:**
- **Code Refactoring**: You must replace all `float` declarations in the C++ code with `ap_fixed` types.
- **Risk of Overflow / Underflow**: You must mathematically prove the maximum possible value your PID error or IK angles could ever reach to ensure you assign enough integer bits. If you assign too few fractional bits, your PID loop will suffer from quantization errors and become unstable.
