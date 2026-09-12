# VRI 2026 — Project Instructions

**VRI 2026** is a self-contained Audio-Visual-Language Action (VLA) demonstration platform built around a 3-DOF parallel manipulator ball-balancing robot. Full history and rationale for every decision below lives in `docs/PROJECT_LOGBOOK.md` — this file states current truth only; read the logbook for *why*.

## System Constraints (CRITICAL — DO NOT VIOLATE)
- **FPGA BRAM Limit:** 612.5 KB
- **FPGA DSP Limit:** 220 slices
- **Target Architecture:** Shared Encoder Backbone (Option A) — no DDR3 weight streaming.
- **Latency Requirement:** 120Hz control loop (< 8.3ms/frame) — binding on any eventual FPGA-hosted production controller; not binding on the Jetson comparison arms.
- **Platform Dimensions (Physical):** Width = **187.5 mm**, Height = **142.0 mm**. Source of truth — never use old values (e.g. 182.5×147.0).
- **Compute (large-model deployment):** NVIDIA **Jetson AGX Orin 64GB Developer Kit, model p3730** (Ampere, Tensor Cores). This is **not** the 2019 Jetson Nano (~15-40x weaker, no Tensor Cores) — never write "Jetson Nano" for this device; a stale doc did this once and it was wrong.

## Signal Flow
```
Webcam → ArUco Homography → Warped 128×128 → Shared CNN → Ball (x,y) + Marker Locations
Microphone → Audio Classifier → Target Colour Command
         ↓
Fusion Module → Motion Command (FORWARD/LEFT/RIGHT/BACKWARD/HOLD/STOP)
         ↓
STM32 / FPGA → PID or RL policy → Inverse Kinematics → Stepper Motor Pulses
```

## Coordinate Contract (hard rule)
Python owns ALL coordinate transforms. Raw camera pixels MUST be converted to physical millimetres before anything is sent to the STM32/FPGA — hardware only ever receives final `(x_mm, y_mm)`. Never push pixel space, homography matrices, or px-per-mm scale factors across that boundary.

## Architecture Decisions (LOCKED — do not change without explicit user instruction)

| Decision | Choice |
|----------|--------|
| Vision CNN | **Shared Encoder Backbone** — one CNN, two heads (Ball + Markers) |
| No. of parameters | **~70K total** — fits 100% in FPGA BRAM |
| Ball head | Heatmap → spatial softmax → soft-argmax (X, Y) |
| Marker head | U-Net-style decoder → binary segmentation mask + heatmap |
| CNN input resolution | **128×128** (downsampled from the ArUco-warped top-down view) |
| Coordinate mapping | **ArUco homography → warped → pixel × scale = mm** (no MLP) |
| Color classification | **Static HSV bin thresholds** first; SVM fallback if <90% accuracy |
| Deployment target | **Host PC (Python + ONNX)** first; then FPGA Verilog via CodeV MCP |
| YOLOv8-nano/ResNet ban | **Lifted 2026-08-13** — was FPGA-budget-driven, no longer binding now that FPGA vision inference is paused. Which vision-model class ("small" Shared Backbone vs. "medium" YOLO/ResNet lineage) backs the expert comparison arm is an **open, undecided question** — don't assume either side. |

## Current State

| Module | Status | Owner |
|--------|--------|-------|
| Vision (Ball + Markers) | Premier pipeline (ArUco + Shared Backbone CNN + marker classifier) working; two legitimate vision-model classes (small/medium) still open, not a gate | `.claude/agents/ml-vision.md` |
| Audio | Root-caused into 3 distinct failure clusters (background leakage, corrupted-clip class confusion, genuine feature overlap); actively being fixed, not "needs debug" in the abstract anymore | `.claude/agents/ml-audio.md` |
| Fusion | Not started — blocked on working Vision (markers) + verified Audio | orchestration-level, no dedicated agent yet |
| FPGA | Reframed 2026-08-13: no longer targeting on-chip vision inference. New role: digital↔optical bridge for the photonic comparison arm (blocked, no interface spec yet) + PID/IK HLS core wiring (live priority, unblocked). Camera→UDP video streaming still separately unresolved. | `.claude/agents/fpga.md` |
| Multimodal / VLA | **The project's major goal as of 2026-08-14.** 3-arm comparison (expert pipeline / large Qwen-derived model on Jetson / photonic platform), all converging on the Jetson AGX Orin as the shared hardware platform for arms 1-2. | `.claude/agents/ml-multimodal.md` |
| Hardware (Tripod, Power) | Not started | User / Electrical Engineer |

## Hardware Decisions (OPEN — human sign-off required)

Agents must not choose, implement, or design around any of these — surface trade-offs and ask the user.

| Open Decision | Options on the table |
| --- | --- |
| Power connector | Continue with existing coaxial DC connector, **or** switch to a custom supply with its own DC connector |
| FPGA power delivery | Regulated pin/rail off the main 24V 6A supply — exact regulator, tap point, isolation all undecided |
| Microphone integration | Standalone INMP441 breakout, **or** design the mic circuit directly into a custom PCB |
| PCB consolidation | Not committed — one PCB carrying motor-driver headers, FPGA comms, mic circuit, and internal PSU is a brainstorm, not a spec |

## Coding Conventions & Engineering Standards

- **Python environment:** `C:/Users/Admin/.conda/envs/ball_balance_env/python.exe`. Always use this interpreter (several pinned deps — tensorflow, torch, fastmcp — are only validated against Python 3.10 in this env).
- **Import pattern:** relative `src.` imports from `host_software/src/` for shared receivers/utils.
- **ONNX model path pattern:** `host_software/ml_vision/models/<model_name>/<checkpoint>.onnx`.
- **Data tier convention:** `data/01_bronze/` raw sessions → `data/02_silver/` processed/labeled pairs → `data/03_gold/` curated train/eval splits, kept physically separate. See the `data-processing-pipeline` skill for the full contract and per-pipeline specifics.
- **Non-blocking execution:** vision and audio inference must never block the frame-capture/control loop — the STM32/FPGA must never receive a stalled coordinate update because of it.
- **Type hints:** all new Python functions require strict type hints.
- **Dependencies:** any new `pip install` must be reflected in both `environment.yml` and `requirements.txt`.

## Repository Structure (pointers — detail lives with the owning agent)

- `host_software/ml_vision/` — vision. Subdirectory rules (`core/`, `data_processing/`, `training/`, `tests/`, `evaluations/`, `experiments/`, `models/`) documented in `.claude/agents/ml-vision.md`.
- `host_software/ml_audio/` — audio. See `.claude/agents/ml-audio.md`.
- `host_software/ml_multimodal/`, `host_software/ml_jetson_vla/` — the 3-arm VLA comparison. See `.claude/agents/ml-multimodal.md`.
- `fpga/` — ZedBoard (Zynq XC7Z020), Vitis/Vivado 2025.2. See `.claude/agents/fpga.md`.
- `firmware/stm32_ml_control_and_vision/BallBalancingBot/` — **current source of truth for STM32 firmware** (RL control net + `SerialCoords.cpp` external-host coordinate input). `firmware/stm32_jetson_remote_control/BallBalancingBot/` is an **intentional**, separate deployment for the Jetson arm (accepts pre-computed step targets via `RemoteStepControl.cpp` instead of running its own inference) — don't confuse it with a stale duplicate.
- **⚠️ Unresolved firmware duplication risk**: at least six directories contain a `BallBalancingBot.ino`/firmware copy (`firmware/BallBalancingBot/`, `firmware/stm32_ml_control_and_vision/BallBalancingBot/` [current], `firmware/ExpertEvaluationFirmware/`, `host_software/ml_control/stm32_ctrl/BallBalancingBot/`, `ball-balancing-bot/BallBalancingBot/`, `host_software/ml_endtoend/firmware/BallBalancingBot/`). Only the current one is confirmed live; the rest have not been individually audited for dead-duplicate vs. legacy-snapshot vs. distinct-purpose. This has already caused one wrong recommendation before being caught — never assume a firmware directory's purpose from its name alone.
- `hardware/platform_templates/` — procedural LaTeX calibration sheets + `ground_truth_manifest.json` (source of truth for marker physical coordinates).
- `docs/plans/` — cross-cutting architecture/implementation plans. **Always read the relevant plan before writing code** (e.g. `implementation_plan_shared_backbone_cnn.md`, `ml_system_parameter_budget.md`).
- `docs/PROJECT_LOGBOOK.md` — the append-only history and rationale behind every decision above. Read it for "why"; don't re-derive already-root-caused bugs from scratch.
- **Deprecated/off-limits (read-only reference, do not add new files):** `host_software/experimental_variants/`, `host_software/new_vla_files/`, `host_software/ml_endtoend/`.

## Agent Capabilities (MCP)

- **`ask_gemini_context`** (`VRI_2026_AI_Router` MCP server): mandatory for reviewing large files, auditing another agent's output, parsing telemetry logs, or reading C++ firmware — don't load large files wholesale to check someone else's work when this exists.
- **`generate_verilog`**: **NOT CURRENTLY AVAILABLE** — this machine lacks the local compute to run the CodeV-DS-6.7B model. Infrastructure stays wired for when that capacity exists (Phase 5 FPGA work); do not invoke until told the compute constraint is resolved.
- Requires a `GEMINI_API_KEY` in `.env` at the repo root (free tier at aistudio.google.com/apikey).

## Sub-agents

Delegate domain work to these rather than working outside your lane:

- **`fpga`** — Verilog/HLS/Vitis/Vivado work in `fpga/`.
- **`ml-vision`** — vision pipeline work in `host_software/ml_vision/`.
- **`ml-audio`** — audio classifier work in `host_software/ml_audio/`.
- **`ml-multimodal`** — VLA comparison work in `host_software/ml_multimodal/` and `host_software/ml_jetson_vla/`.

Boundary rule: none of these should modify shared core utilities (`host_software/src/`) or firmware without explicit orchestration — coordinate outputs from Vision must match exactly what the Audio/Fusion state machine expects.

## Skills

Reach for these when the task matches, rather than re-deriving the checklist from scratch:

- **`fpga-pipeline`** — writing/modifying FPGA RTL or HLS code: verification-first sequence, clocking/CDC pitfalls, hls4ml layer-support checks.
- **`data-pipeline-verification`** — before trusting a new/changed data-processing script's output.
- **`dataset-integrity-check`** — auditing an existing dataset at rest (leakage, corruption, coverage, duplication).
- **`model-iteration-constraints`** — designing a new model or a structural change to an existing one.
- **`data-processing-pipeline`** — building/extending a raw-capture-to-labeled-data pipeline (shared Medallion contract + per-module ground-truth method).
