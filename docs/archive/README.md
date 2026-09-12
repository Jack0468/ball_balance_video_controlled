# Archive

Superseded or purely historical documents, kept for reference — not current design.
Each entry below is a pointer to the doc that actually replaced it, so a reader isn't left
guessing whether something here is safe to follow.

| File | Was | Superseded by |
|---|---|---|
| `SYSTEM_ARCHITECTURE.md` | Macro-architecture for an Opal Kelly XEM3010 (Spartan-3) FPGA doing on-chip vision inference | `CLAUDE.md` Current State table (ZedBoard, FPGA reframed 2026-08-13 to bridge+PID/IK role) + `docs/plans/ml_system_parameter_budget.md` |
| `poster_draft.md` | Competition poster draft describing a YOLOv8/ResNet + Wav2Vec expert system vs. VLA comparison | `CLAUDE.md` (Shared Backbone CNN is the locked vision architecture) + `host_software/ml_jetson_vla/docs/ARCHITECTURE.md` (current 3-arm Jetson/photonic comparison) |
| `HLS_DATA_TYPES.md` | Early, general float-vs-fixed-point cost/benefit discussion for HLS | `docs/plans/ml_system_parameter_budget.md` §5.7-5.8 (real trial data, calibration methodology, per-layer results) |
| `IMPLEMENTATION_GUIDE.md` | Step-by-step setup guide referencing YOLO/MobileNet SSD benchmarking, the legacy `ball-balancing-bot/` C++ codebase, and a Teensy-era workflow | `README.md` + `CLAUDE.md` (current onboarding, current Shared Backbone CNN, current STM32 firmware) |
| `VITIS_TO_ISE_GUIDE.md` | Hybrid Vitis HLS + legacy Xilinx ISE 14.7 workflow required only by the Opal Kelly XEM3010 (Spartan-3) | `docs/HARDWARE_AND_SOFTWARE_PREREQUISITES.md` (ZedBoard uses Vitis/Vivado 2025.2 natively, no ISE hybrid needed) |
| `ENGINEERING_STANDARDS.md` | Engineering standards for a Teensy-based "baseline revision" | `CLAUDE.md`'s "Coding Conventions & Engineering Standards" and "Coordinate Contract" sections (the Python-side standards were carried forward near-verbatim; the Teensy baseline itself is gone, current MCU is STM32) |
| `TODO.md` | Old chronological task list, Phase 1-era | `docs/PROJECT_LOGBOOK.md` (current append-only history) |
| `ml_vision_todo.md` | Old vision-pipeline task list, pre-hardware planning | `docs/PROJECT_LOGBOOK.md` + `.claude/agents/ml-vision.md` |

See `CLAUDE.md`'s "Terminology: Phase / Track / Arm" section for how the current project's
workstreams are named — several of the archived docs above use an informal, inconsistent
"Phase N" numbering of their own that does not match the project-level Phase numbering used
today; don't cross-reference the two.
