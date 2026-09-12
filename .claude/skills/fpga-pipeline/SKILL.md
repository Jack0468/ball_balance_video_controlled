---
name: fpga-pipeline
description: Use when writing or modifying FPGA/RTL/HLS code for the ZedBoard (Verilog, Vitis HLS, hls4ml, Vivado IP packaging, VDMA, clock-domain crossing) — a verification-first sequence plus hardware pitfalls mined from this project's real, already-paid-for debugging history. Load before writing new RTL/HLS, not after something breaks.
---

# FPGA code-writing pipeline

This project has repeatedly paid multi-day debugging costs for FPGA bugs that a testbench or a known pitfall check would have caught immediately. This skill exists so those costs aren't paid twice.

## 1. Check documentation ground truth before trusting any FPGA doc

Current: `docs/PROJECT_LOGBOOK.md`, `docs/HARDWARE_AND_SOFTWARE_PREREQUISITES.md`, `docs/udp_research_guide.md`, `fpga/docs/*.md`. Legacy/abandoned (Opal Kelly XEM3010 era, historical only): `docs/SYSTEM_ARCHITECTURE.md`, `docs/IMPLEMENTATION_GUIDE.md`, `docs/VITIS_TO_ISE_GUIDE.md`, `fpga/main_controller/`, `fpga/camera_i2c/legacy_xem3010/`. `docs/HLS_DATA_TYPES.md` is toolchain-agnostic and still applies despite living near the stale docs.

## 2. Vitis/HLS 2025.2 syntax caution

LLM training data is unreliable on the current Vitis Unified IDE workflow and HLS pragma/syntax conventions. Do not confidently assert IDE steps, pragma syntax, or tool-specific behavior from memory — say so explicitly and ask for confirmation against real tool output or official Xilinx UG/PG docs. A wrong confident guess costs more debugging time than an honest "I'm not sure."

## 3. Verify digitally before touching physical hardware — always, in this order

1. Write and run a testbench with known inputs → verified outputs, entirely in simulation. Every real hardware regression in this project's history that reached physical hardware first cost multiple debugging sessions to trace back to a root cause a testbench would have caught in minutes.
2. Verify sensor/camera input in isolation (VGA/monitor sanity check, or route a single frame via UART/UDP for analysis) before wiring it into the full pipeline.
3. Verify actuation in isolation (e.g. a 90° rotation test, return to zero) before closing the loop.
4. Verify any new peripheral (mic, sensor) at the raw-data level (store to RAM, transfer, inspect) before running inference/control against it live.
5. Only once 1-4 pass do you close the full loop on real hardware.

## 4. Clock-domain and timing pitfalls (each of these has already caused a real, multi-day bug in this project — check for it explicitly)

- **Route external pixel/data clocks only to dedicated GCLK pins.** Never assign a BUFG-driven net directly to an output pin — Spartan-3-class silicon forbids it and the router will silently leave the pin floating (reads as noise on a scope, not an error).
- **Cross every async signal into a different clock domain's state machine through a dual-rank flip-flop synchronizer** — a bare async reset (or any control signal) driving a fast synchronous state machine directly causes metastability lockups. This has happened more than once in this project from a signal that "should" have been synchronized but wasn't.
- **Verify PLL VCO frequency against the chip's documented operating range** before trusting a multiplier/divider calculation — a request outside the valid range (e.g. below the chip's minimum) can silently fail to lock rather than erroring loudly, producing an unstable or wrong-frequency output that looks like a different bug.
- **Verify physical pin-to-PLL-output (or pin-to-peripheral) mapping against the board's actual schematic/manual** — do not assume a numbering convention. Two separate real incidents in this project came from trusting an assumed mapping instead of checking the manual.
- **A `busy`/protect signal in any memory controller or arbiter must cover ALL non-idle states** (including INIT/REFRESH-style housekeeping states), not just the obvious READ/WRITE states — otherwise an arbiter will issue commands during a state it thinks is safe but isn't, and the controller silently drops them.
- **Avoid `#`-delay intra-assignment continuous drives inside a clocked `always` block that execute every cycle** — this can trap an event-driven simulator (e.g. ISim) in a zero-delay loop. Drive shared/bidirectional (`inout`) buses via pure combinational `assign`, not delayed non-blocking assignments.
- **A camera or peripheral clock line at unexpectedly low/unstable voltage or amplitude on a scope is a real signal-integrity finding, not sensor error** — check drive strength and capacitive loading before assuming the logic is wrong.

## 5. HLS / hls4ml layer-support check — before committing to an architecture for hardware

- Check hls4ml's actual converter source for native layer support before assuming it. As of this project's last audit: `Conv2d`, `BatchNorm2d`, `MaxPool2d`, `Linear`, `Upsample` are native; `GELU`, `ConvTranspose2d`, `AdaptiveAvgPool2d` are not (verify against the current hls4ml version, this can change). A missing layer needs a custom Extension API kernel — budget for this explicitly (it's a bounded, small effort per hls4ml's own examples, not open-ended research), don't discover it as a surprise blocker after training.
- Fold `BatchNorm2d` into the preceding conv's weights/bias at export time — never implement it as a separate runtime op on-chip.
- Drop any `interpolate`/resize call that is a mathematical no-op given a fixed, locked input resolution — don't synthesize it.
- **Activation buffering, not weight storage, is usually the real BRAM bottleneck for CNN-shaped models** — a single early-layer whole-frame feature map can dwarf all the weights combined. Design a streaming, line-buffered dataflow (row buffers per conv layer, Xilinx Vitis Vision Library pattern), never translate PyTorch layers into whole-frame on-chip arrays.
- **`ap_fixed<W,I>` bit-width choices must be calibrated from real observed per-layer activation/weight ranges**, not a blanket guessed config. A low-bit-width result that "collapses" (large error jump) is more often uncalibrated range saturation than proof that bit-width is infeasible — re-derive `int_bits` from actual observed data ranges before concluding a precision level doesn't work, and re-run any earlier quantization trial if the checkpoint it used is later found to have a bug (its numbers are invalidated, not just outdated).

## 6. Reuse the established weight-compiler pattern

Before writing a new PyTorch/Keras-to-hardware weight exporter from scratch, read the existing `--hls` mode of whichever exporter script already exists in this repo for the pattern (state_dict/checkpoint → C header with shape macros) — don't reimplement it, and don't modify a working exporter outside your own module's territory.

## 7. Repository discipline

Keep current (Zynq) and legacy (Opal Kelly) code visibly separated — don't let stray root-level logs or duplicate doc copies bleed into new work. Write new FPGA docs only under `fpga/docs/`. Flag doc drift you notice rather than silently fixing docs outside your task.
