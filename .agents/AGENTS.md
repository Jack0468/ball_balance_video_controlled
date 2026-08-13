# VRI 2026 AI Agent Rules

## System Constraints (CRITICAL — DO NOT VIOLATE)
- **FPGA BRAM Limit:** 612.5 KB
- **FPGA DSP Limit:** 220 slices
- **Target Architecture:** Shared Encoder Backbone (Option A) — No DDR3 weight streaming.
- **Latency Requirement:** 120Hz control loop (< 8.3ms per frame).
- **Platform Dimensions (Physical):** Width = **187.5 mm**, Height = **142.0 mm**. These are the source of truth. Never use old values (e.g., 182.5×147.0).

---

## Project Overview

**VRI 2026** is a self-contained Audio-Visual-Language Action (VLA) demonstration platform built around a 3-DOF parallel manipulator ball-balancing robot.

### Demo Objective (Intern Task 1: Audio–Visual Fusion)
A spoken command (e.g., "Go to blue") causes the system to:
1. Identify the physical location of the blue target marker using vision.
2. Compute a motion command: `FORWARD / LEFT / RIGHT / BACKWARD / HOLD / STOP`.
3. Transmit that command to the STM32/FPGA to physically drive the ball to the target.

**Ablation behaviour required:**
- **Audio + Video**: Identifies the correct target and reaches it.
- **Audio only**: Knows which colour was requested but not its location.
- **Video only**: Knows where all markers are but not which was requested.

### System Signal Flow
```
Webcam → ArUco Homography → Warped 128×128 → Shared CNN → Ball (x,y) + Marker Locations
Microphone → Audio Classifier → Target Colour Command
         ↓
Fusion Module → Motion Command (FORWARD/LEFT/RIGHT/BACKWARD/HOLD/STOP)
         ↓
STM32 / FPGA → PID → Inverse Kinematics → Stepper Motor Pulses
```

### Hardware Components
- **Platform:** 3-DOF parallel manipulator (187.5 × 142.0 mm), 3× NEMA 17 stepper motors, TMC2208 drivers.
- **Controller (Current):** STM32 running PID + IK firmware (`firmware/BallBalancingBot/`).
- **Controller (Target):** ZedBoard FPGA (Xilinx XC7Z020) — all ML inference + PID in hardware.
- **Camera:** Overhead webcam, 640×480.
- **Microphone (Planned):** INMP441 digital I2S mic (3.3V, cheap, 16kHz). Note: the audio *data* interface is I2S, not I2C — some planning notes say "I2C" by mistake (easy to confuse the acronyms); do not "correct" this back to I2C.
- **Compute (Large-Model Deployment):** NVIDIA **Jetson Orin Nano, 8GB** (confirmed 2026-08-13 — Ampere GPU w/ Tensor Cores, 20-67 TOPS INT8, NOT the original 2019 Maxwell-based Jetson Nano, which is ~15-40x weaker and has no Tensor Cores). Target device for large multimodal/VLA models that are out of scope for both the FPGA (612.5 KB BRAM / 220 DSP budget) and the STM32. Not used for the lightweight expert-side vision/audio/control models. Realistic capability ceiling: INT4-quantized models up to ~4B params (e.g. Qwen2.5-VL-3B) at a few tokens/sec; NOT enough headroom for 7B-class models like OpenVLA (~2Hz even on the much stronger Jetson AGX Orin). See Multimodal/VLA section below for what this rules in/out.

---

## Current State of the Project

| Module | Status | Notes |
|--------|--------|-------|
| Vision (Ball) | ✅ Working | Premier pipeline: ArUco homography + 2D CNN (`cnn_2d_tracker_0730_v3`) + MLP corrector |
| Vision (Markers) | 🔧 In Progress | **Shared Backbone CNN planned** — see `implementation_plan_shared_backbone_cnn.md` |
| Audio | ⚠️ Needs Debug | Produces incorrect outputs during concurrent robot operation; matched filter suspected |
| Fusion | ⬜ Not Started | Requires working vision (markers) + verified audio |
| FPGA Port | 🔄 Reframed 2026-08-13 | ZedBoard/Zynq (XC7Z020), Vitis/Vivado 2025.2. **No longer targeting on-chip vision-CNN inference** — that work (hls4ml GELU LUT, Upsample+Conv2d decoder reconstruction) is paused, not deleted. New role: digital↔optical signal bridge for the photonic computing comparison arm (see Multimodal/VLA row below) — **blocked**, no interface spec exists yet for the optical platform. Camera→UDP video streaming is separately still unresolved, independent of this pivot. See `.agents/agent_fpga.md` |
| Multimodal / VLA | 🔧 In Progress | Reframed 2026-08-13 into a **3-arm comparison**: (1) the expert pipeline (vision + audio + control), (2) a large (billions-of-param) Qwen-derived model adapted from the lab partner's work, deployed standalone on the Jetson Orin Nano, (3) a model compiled for a photonic/optical computing platform, bridged via the FPGA (see above). Expert pipeline should also deploy to the Jetson (not just laptop) to remove hardware-platform as a confound between arms 1 and 2 — agreed direction, not yet implemented. `RT1LiteVLA` (the in-house custom baseline) remains a secondary/optional comparison point, not arm 2. See `.agents/agent_ml_multimodal.md` and `host_software/ml_jetson_vla/docs/ARCHITECTURE.md` |
| Hardware (Tripod) | ⬜ Not Started | Camera/mic mount design needed |
| Hardware (Power) | ⬜ Not Started | 24V 6A supply, FPGA power, PCB consolidation TBD |

---

## External / Comparative Context

- A lab partner is independently building a comparable model for this same task using **Qwen** (a general-purpose multimodal/language model) as their VLA approach. This is useful framing for our own general-purpose baseline and a potential future cross-comparison point, but it is not a dependency: do not assume access to their model/weights/code, and do not adopt Qwen as our architecture without explicit instruction from the user.
- **`RT1LiteVLA` is not actually a "general-purpose" VLA (2026-08-13 finding):** it deploys trivially on the Jetson Orin Nano (only ~12M params, dwarfed by its budget) — capacity is not the deployment problem. The problem is generality: it uses a hardcoded 5-word closed vocabulary embedding table (not real language understanding), a frozen off-the-shelf ImageNet-pretrained ResNet18 (not pretrained on robot/multimodal data at any real scale), and its Stage-2 RL fine-tuning (`train_vla.py`'s `fine_tune_rl()`) is currently a stub that only prints — it does not actually run PPO/REINFORCE. Treat `RT1LiteVLA` as a cheap in-house **lightweight custom baseline**, not as the "general-purpose VLA" arm of the expert-vs-VLA comparison — it is architecturally closer to a third category than to real general-purpose models like Qwen-VL/OpenVLA/Octo.
- **Deployment feasibility survey for a genuine general-purpose VLA/VLM on the confirmed 8GB Jetson Orin Nano:**
  - **Qwen2.5-VL-3B (INT4-quantized via MLC/TensorRT-LLM):** NVIDIA's own Jetson AI Lab benchmarks target this exact class of model on the Orin Nano 8GB (models up to ~4B params). Realistic throughput is a few tokens/sec — fine as a discrete-command planner (image + instruction → target color/action token, same shape as our existing audio-command output) but not for direct per-frame motor-torque generation at any meaningful control rate.
  - **Octo-small (~93M params, purpose-built VLA, not VLM-based):** the closest architectural match to our expert-vs-VLA framing (real robot-action output head, not text tokens). Published numbers: ~5-8Hz on the much stronger Jetson AGX Orin, ~20Hz on an RTX 4090 — expect low single-digit Hz on the weaker Orin Nano. Comparable published on-device lightweight-VLA work (`LiteVLA-Edge`) reports ~6.6Hz on similar-class edge hardware, which is a useful ballpark.
  - **OpenVLA (7B): ruled out.** Even on the far stronger Jetson AGX Orin it only reaches ~2Hz, and 7B params leaves little headroom in the Orin Nano's 8GB shared CPU/GPU memory once the OS and other processes are accounted for. Do not attempt to deploy this to the Orin Nano.
  - Decision on which path (Qwen2.5-VL-3B-as-planner vs. Octo-small-as-policy vs. both) is open — surface the tradeoff to the user rather than picking unilaterally, per this doc's Hardware Decisions convention.

---

## ML Component Sizes (Reference Baselines)

| Component | Model | Params | Status |
|-----------|-------|--------|--------|
| Ball Tracker | `cnn_2d_tracker_0730_v3` (BasicCNN) | ~340K | ✅ Working — too large for BRAM alone |
| MLP Corrector | `mlp_corrector_*` | ~70K | ✅ Working — being eliminated in new architecture |
| **Shared Backbone CNN** | `train_cnn_2d_tracker_marker` | **~70K total** | 🔧 Target architecture |
| Audio Classifier | Conv1D × 3 + Dense | ~13.5K | ⚠️ Needs debugging |
| Control Net | Small MLP (9→32→32→3) | ~1.5K | ✅ Working — validated via the resistive touchpad sensor; not yet verified end-to-end with vision-derived coordinates |
| VLA (RT1LiteVLA) | ResNet18 + Transformer | ~12M+ | 🔧 Not for FPGA — host PC for training/dev, **Jetson Nano** is the target deployment device; compared against the baseline expert pipeline, see `.agents/agent_ml_multimodal.md` |

> [!IMPORTANT]
> The YOLOv8-nano/ResNet parameter-budget ban is **no longer FPGA-driven as of 2026-08-13** — the FPGA is not currently targeting on-chip vision inference at all (see Current State table above). Do not cite "too large for FPGA" as a reason to avoid these models going forward; see the open question below for the actual live consideration.

### Open question: which vision model backs the "expert" comparison arm? (2026-08-13, NOT decided)

The current `run_eval_expert.py` reference implementation actually uses the **older** `yolov8_platform_pose_markers_iphone_v1` + `mlp_corrector_iphone_v1` pipeline, not the Shared Backbone CNN — the tiny-parameter design was built as a future FPGA target, not what the expert arm currently runs. Now that FPGA vision inference is paused, parameter count is no longer a constraint either way (both the old pipeline and the Shared Backbone CNN are trivially small for a Jetson Nano or even a laptop).

**The real lurking variable is training-data recency, not size.** The YOLO/ResNet/CNN lineup (tags `_iphone_`, `_0728_`, `_0730_`) was never trained on Dataset 8/9 — the merged, bug-fixed, synthetic-augmented dataset (52,772 real + 59,336 synthetic rows) that only `shared_vision_backbone_v1` has seen. Comparing the two pipelines' accuracy as-is would conflate "architecture" with "which one saw better data," not isolate either variable cleanly.

**Decision:** deliberately left open, pending the Shared Backbone CNN's evaluation results once the current training run finishes. Do not pick a side in either direction (don't assume "revert to the bigger models" and don't assume "the new one wins") until that evaluation exists. A clean architecture-only ablation (does model size matter, holding data constant) would require retraining the YOLO/ResNet lineage on Dataset 8/9 first — flagged as a low-priority "for fun" follow-up, not a blocker on the actual decision, and not worth the time investment unless there's spare capacity for it.

---

## Architecture Decisions (LOCKED)

These decisions have been made and MUST NOT be changed without explicit user instruction:

| Decision | Choice |
|----------|--------|
| Vision CNN | **Shared Encoder Backbone** — one CNN, two heads (Ball + Markers) |
| No. of parameters | **~70K total** — fits 100% in FPGA BRAM |
| Ball head | Heatmap → spatial softmax → soft-argmax (X, Y) |
| Marker head | U-Net decoder → binary segmentation mask + heatmap |
| CNN input resolution | **128×128** (downsampled from 500×500 ArUco-warped top-down view) |
| Coordinate mapping | **ArUco homography → warped → pixel × scale = mm** (no MLP) |
| Color classification | **Static HSV bin thresholds** first; SVM fallback if <90% accuracy |
| Deployment target | **Host PC (Python + ONNX)** first; then FPGA Verilog via CodeV MCP |

---

## Hardware Decisions (OPEN — Human Sign-off Required)

Unlike the software architecture decisions above, these are **not settled**. Agents must not choose, implement, or design around a specific option here — surface the trade-offs and ask the user. Do not silently assume one option while writing code, BOM lists, or CAD.

| Open Decision | Options on the table |
| --- | --- |
| Power connector | Continue with existing coaxial DC connector, **or** switch to a custom supply with its own DC connector |
| FPGA power delivery | Power the ZedBoard via a regulated pin/rail from the main 24V 6A supply — exact regulator, tap point, and any other considerations (inrush, isolation, etc.) are undecided |
| Microphone integration | Standalone INMP441 breakout module, **or** design the mic circuit directly into a custom PCB from the outset |
| PCB consolidation ("robustness" idea) | Not committed. Idea floated: one PCB carrying motor driver headers (not integrated drivers), FPGA communication/logic, the audio/mic circuit, and an internal power supply/transformer so the only external connection is an IEC mains cable. Treat as a brainstorm to revisit, not a spec. |

---

## Data & Training Rules

- **Data collection** uses the existing `firmware/BallBalancingBot` (random ball movement) with `host_software/data_collection/collect_webcam_data.py`. Do NOT propose new firmware for this.
- **4 data sessions must be recorded**, one per platform configuration. These sheets already exist in `hardware/platform_templates/` — do not redesign or add new marker sets without explicit instruction:
  1. `aruco_markers_00.tex` — blank platform (ArUco corner markers only) — used for synthetic marker compositing
  2. `aruco_markers_01.tex` — 5 solid-colour circles: blue, black, red, green, yellow
  3. `aruco_markers_02.tex` — 4 blue shapes: circle, triangle, square, hexagon
  4. `aruco_markers_03.tex` — 5 mixed shape/colour markers: black circle, blue triangle, yellow square, green hexagon, red triangle
- Marker centre points are known by design (hardcoded in each sheet's `_manifest.json` from the LaTeX layout) and are mapped to ground truth post-collection via homography — never derived by colour tracking.
- **Training data split:** 60% synthetic composites (blank session) + 40% real printed sheets (01–03). Synthetic composites also vary shape and color (not just position), deliberately biased toward the 9 shape×color combinations absent from the real printed sheets (real sheets cover only 11/20 possible combos — see `dataset_info.md` Dataset 9), to teach appearance generalization, not just position invariance. Implemented in `generate_synthetic_marker_composites.py` + `combine_shared_vision_training_mix.py`.
- **Evaluation is ONLY run on new, unseen sheets** (e.g., `aruco_markers_04.tex`). Never evaluate on training sheets. `aruco_markers_04.tex` has not been printed/collected as of 2026-08-12 — until it exists, evaluation is run on the held-out temporal slice of the existing all-real Dataset 8 (`host_software/data/03_gold/shared_vision/`) instead. This is same-sheet/different-frames evaluation (disjoint rows within sessions 01–03, which also feed training), not a fully unseen sheet — strictly weaker evidence. Do not let training data (synthetic or real) leak into that CSV; keep it physically separate from whatever training-mix CSV is used (see `03_gold/shared_vision_synthetic_mix/`).
- **Ball labels** are generated by **backward-mapping** telemetry `{pitch, roll}` → physical mm → ArUco homography → pixel, via `auto_label_shared_vision.py`. Do NOT use color tracking to label the ball.
- **Marker labels** are generated by projecting `ground_truth_manifest.json` physical coordinates through the ArUco homography to produce binary masks.
- **Platform dimensions** in all scripts must use `PLATFORM_W = 187.5` and `PLATFORM_H = 142.0`.

---

## Repository Structure Rules

All code additions MUST be placed in the correct directory. Never dump scripts into a generic `scripts/` folder.

### 1. ML Vision (`host_software/ml_vision/`)
- `core/`: Classical CV + deterministic algorithms. Key files: `preprocessor.py`, `coordinate_math.py`, `marker_classifier.py`.
- `data_processing/`: Auto-labeling, synthetic compositing, telemetry backward-mapping. Key files: `auto_label_shared_vision.py`.
- `training/`: CNN models, PyTorch Datasets, ONNX export. **Naming convention: `train_cnn_2d_tracker_marker.py`**. No YOLO scripts in new work.
- `tests/`: Functional validation and unit tests only. Not for ML metrics or benchmarks.
- `evaluations/`: Metric generation (IoU, centroid error mm, confusion matrices, latency benchmarks).
- `experiments/`: One-off comparative trials (e.g. augmentation sweeps) producing a comparison report, not a single pass/fail or a fixed benchmark. Distinct from `tests/` (unit tests) and `evaluations/` (fixed metric generation).
- `models/`: Saved `.onnx` and `.pt` model weights. Organised by model name/version subdirectory.

### 2. Hardware & Firmware
- `hardware/platform_templates/`: Procedural LaTeX sheets and `ground_truth_manifest.json` (source of truth for marker physical coordinates).
- `firmware/BallBalancingBot/`: Single source of truth for STM32 telemetry, motor control, and PID. Do not add new firmware sketches for data collection.
- `host_software/data_collection/`: Python scripts for webcam and STM32 interaction. Key files: `collect_webcam_data.py`, `sync_webcam_telemetry.py`.

### 3. Documentation
- `docs/plans/`: Cross-cutting architecture and implementation plans. **Always read the relevant plan before writing code.**
  - `implementation_plan_shared_backbone_cnn.md` — Active implementation plan for the vision system.
  - `ml_system_parameter_budget.md` — FPGA resource budget. Consult this before designing any model.
- Module-local plans live under `host_software/<module>/docs/plans/`, mirroring the pattern above at module scope. Current example: `host_software/ml_audio/docs/plans/audio_eval_notebook_refactor_plan.md` — the active plan for the audio module's `core/`/`training/`/`evaluations/` refactor and dataset-corruption remediation. Read it before assigning further audio work.

### 4. Deprecated / Off-limits Directories
- `host_software/experimental_variants/`: Legacy experimental scripts. Read for reference only; do not add new files here.
- `host_software/new_vla_files/`: Inactive Arduino/serial firmware (`VLADirectControl`) module. Do not modify — read-only reference for the Multimodal/VLA agent's eval scripts.
- `host_software/ml_audio/`: Active — reactivated under Roadmap Phase 2 ("Audio Verification"). See `.agents/agent_ml_audio.md`.
- `host_software/ml_multimodal/`: Active — the in-house `RT1LiteVLA` baseline and its BC/RL training scaffold, now a secondary/optional arm in the 3-arm comparison (independent of the Vision/Audio/FPGA roadmap phases; does not draw against the FPGA resource budget). See `.agents/agent_ml_multimodal.md`.
- `host_software/ml_jetson_vla/`: Active (added 2026-08-13) — arm 2 of the 3-arm comparison: a large (billions-of-param) Qwen-derived model, adapted from a lab partner's work, deployed standalone on the Jetson Orin Nano. See `host_software/ml_jetson_vla/docs/ARCHITECTURE.md` for full context and open blockers (model access, deployment pipeline).
- `host_software/ml_endtoend/`: Orphaned early end-to-end integration snapshot (committed once, 2026-08-02, never touched since). Not formally deprecated but not part of the active architecture either — read-only reference at most.

### 5. FPGA (`fpga/`)
- Current architecture: ZedBoard (Xilinx Zynq-7000, XC7Z020-1CSG484CES), built with **Vitis/Vivado 2025.2**. Migrated from an Opal Kelly XEM3010 (Spartan-3) board on 21/07/2026 after a fatal clocking limitation (no GCLK pins available for the camera pixel clock) — see `docs/PROJECT_LOGBOOK.md`.
- **Ground-truth docs:** `docs/PROJECT_LOGBOOK.md`, `docs/HARDWARE_AND_SOFTWARE_PREREQUISITES.md`, `docs/udp_research_guide.md`, `fpga/docs/*.md`. **Stale/legacy docs (Opal Kelly-era, historical reference only):** `docs/SYSTEM_ARCHITECTURE.md`, `docs/IMPLEMENTATION_GUIDE.md`, `docs/VITIS_TO_ISE_GUIDE.md`. `docs/HLS_DATA_TYPES.md` is the exception among the older docs — its float-vs-`ap_fixed<W,I>` reasoning is toolchain-agnostic and still applies.
- `fpga/hls_hardware/`: HLS C++ control law (PID + inverse kinematics), using `ap_fixed<32,16>`. Written but **not yet wired into** `fpga/vitis/src/main.c`.
- `fpga/camera_i2c/`, `fpga/zynq_camera_sys/`, `fpga/vitis/src/`: OV7670 → AXI VDMA → UDP camera pipeline. Basic UDP communication (ping-level) has been established before, but a working end-to-end video stream from the FPGA to the laptop has **not** yet been formulated — this is still open, not done. See `docs/udp_research_guide.md` for the current debugging state (six untested hypotheses for the intermittent bare-metal UDP failures).
- `fpga/main_controller/`, `fpga/camera_i2c/legacy_xem3010/`: legacy Opal Kelly Verilog. Reference only, not the current build path.
- See `.agents/agent_fpga.md` for the active FPGA sub-agent's scope and mandate.

---

## Coding Conventions

- **Python environment:** `C:/Users/Admin/.conda/envs/ball_balance_env/python.exe`. Always use this interpreter.
- **Import pattern:** Use relative `src.` imports from `host_software/src/` for shared receivers/utils.
- **ONNX model path pattern:** `host_software/ml_vision/models/<model_name>/<checkpoint>.onnx`.
- **Data tier convention:**
  - `data/01_bronze/` — Raw recorded sessions (video + telemetry CSV).
  - `data/02_silver/` — Processed, labelled pairs (frame + mask/annotation).
  - `data/03_gold/` — Final curated training/evaluation splits.
- **Telemetry sync protocol:** STM32 streams `{pitch, roll, timestamp}` using `0xAABBCCDD` sync header.

---

## Python Engineering Standards

All new Python code MUST adhere to these standards (from `docs/ENGINEERING_STANDARDS.md`):

- **Non-blocking execution:** Vision and audio models must run asynchronously. Audio polling or model inference must never freeze the frame capture loop — otherwise the STM32/FPGA receives stalled coordinate updates.
- **Type hints:** All Python functions must include strict type hints (e.g., `def get_position() -> tuple[float, float]:`).
- **Coordinate translation:** Python is responsible for ALL coordinate transforms. Raw camera pixels MUST be converted to physical mm before being sent to the STM32/FPGA. The hardware must only ever receive final `(x_mm, y_mm)` values.
- **Dependencies:** Any new `pip install` must be reflected in both `environment.yml` and `requirements.txt`.

## Audio Module Reference

**Command Set:** `"go red"`, `"go blue"`, `"go green"`, `"go yellow"`, `"hold"`, `"stop"`

**Known Issue:** The audio classifier produces incorrect outputs during concurrent robot operation. The matched filter for the background category is suspected to be the root cause. Debug sequence:
1. Test live audio inference **without** the robot running.
2. Inject robot operation noise digitally and test by inspection.
3. Scrutinize the Python state machine logic for race conditions.

---

## System Evaluation Metrics

All system-level benchmarks use these four metrics (from `docs/EVALUATION_STRATEGY.md`):

| Metric | Formula / Definition | Target |
|--------|---------------------|--------|
| Steady-State Error | `sqrt((x_ball - x_target)² + (y_ball - y_target)²)` mm | < 10mm |
| Settling Time | Time until ball enters and stays within 20mm radius for 500ms | As fast as possible |
| Control Effort | `Σ |θ(t) - θ(t-1)|` across all 3 motors | Minimise (smooth actuation) |
| Task Success Rate | % of target shifts reached without dropping ball | Maximise |

**Standard evaluation sequence:** 11 commands injected at 10-second intervals: `go_grey → go_blue → go_green → go_yellow → go_red → FORWARD → LEFT → RIGHT → BACKWARD → HOLD → STOP`.

---

## Data Pipeline Gotchas

When processing or generating datasets, be aware of these pipeline rules (from `host_software/ml_vision/docs/DATA_PIPELINE.md`):

- **Frozen-frame filtering:** The resistive touchpad has a 1.5s hardware debounce when the ball is lifted. Any frame where the physical coordinate doesn't fluctuate by ≥ 0.1mm (ADC noise floor) must be discarded. This prevents models from learning phantom static coordinates.
- **Spatial normalisation:** The ball naturally clusters near the platform centre. All training datasets must be downsampled using `normalize_spatial_density.py` (5mm × 5mm grid, outlier-clipped) to remove spatial bias.
- **Label deduplication:** Telemetry runs at 100Hz+; camera runs at 30fps. After sync, deduplicate so there is exactly 1 row per physical video frame.
- **Webcam sync (Pipeline A):** `collect_webcam_data.py` records `rgb_video.mp4` + `frame_timestamps.csv` + `telemetry.csv` simultaneously. Sync via `sync_webcam_telemetry.py` — do NOT try to sync by frame index.

---

## Long-Term Roadmap (Broad Context — Do Not Implement All At Once)

### Phase 1 — Lightweight Vision Pipeline (ACTIVE)
- Complete Shared Backbone CNN (ball + marker detection) and evaluate on host PC.
- See `docs/plans/implementation_plan_shared_backbone_cnn.md` for the full plan.

### Phase 2 — Audio Verification
- Troubleshoot audio model during live concurrent robot operation.
- Verify matched filter is correctly reducing false negatives for the background category.
- Test live audio inference in isolation first, then add robot operation noise digitally.

### Phase 3 — Full System Evaluation on CPU
- Run end-to-end latency measurements (vision + audio + fusion + STM32 round-trip).
- Measure latency in multiple forms — compare across the different vision pipeline variants (see `host_software/experimental_variants/`, read-only reference), not just the premier pipeline, to document the accuracy/latency trade-off space.
- Generate evaluation metrics report for CPU implementation before FPGA port.

### Phase 4 — Hardware (Tripod & Power)
- Design 3D-printed camera + microphone mount (tripod). Consider angle calibration — either measurements routed/carved into the print for direct angle reading, or a hard-stop design.
- Route cables for signal integrity (short as possible). FPGA should sit at the base for weight/mass-distribution stability, not hanging off the side.
- Power supply and mic-integration approach: see **Hardware Decisions (OPEN)** above — do not assume an option while designing.

### Phase 4.5 — Additional Angle-Specific Data Collection (conditional)
- Only needed if the camera → FPGA → UDP → laptop pipeline proves difficult once the tripod/mount is in place.
- Goal: collect training data from the actual mounted camera angle.
- Requires solving video transport off the FPGA — either UDP streaming over the FPGA's Ethernet port, or an STM32-buffered stream into the PC as a fallback. Neither is chosen yet.

### Phase 5 — FPGA Port (ACTIVE, REFRAMED 2026-08-13 — owned by `.agents/agent_fpga.md`)
- Target: ZedBoard (Xilinx XC7Z020-1CSG484CES), Vitis/Vivado **2025.2**. See `AGENTS.md` §"Repository Structure Rules → 5. FPGA" for which docs are current vs. legacy (Opal Kelly-era) ground truth.
- **The FPGA's job changed 2026-08-13: it is no longer targeting on-chip vision-CNN inference.** The prior on-chip vision plan (weight compilation, hls4ml GELU LUT, Upsample+Conv2d decoder reconstruction — all real, verified engineering) is **paused, not deleted** — see `docs/plans/ml_system_parameter_budget.md`'s status note. It may become relevant again if the architecture below changes.
- **New role: digital↔optical signal bridge** for a photonic/optical computing platform that runs a model "compiled from" a pretrained VLA/LLM (see Phase 6 below) — this is now **blocked**: no interface spec (electrical/protocol/timing) for the optical platform exists yet. Do not write real Verilog against a guessed protocol; scope this as architecture sketching only until a real spec is available.
- Video transport off the FPGA is **still separately not solved**: basic UDP communication has been established before, but a working camera→FPGA→UDP→laptop video stream has not been formulated. See `docs/udp_research_guide.md` for the current debugging state. This is independent of the optical-bridge pivot — the FPGA still needs working sensor I/O regardless of which downstream role it plays.
- The already-written `fpga/hls_hardware/` PID+IK HLS core is still not wired into the live application (`fpga/vitis/src/main.c` currently only does camera-stream UDP, with even VDMA commented out for a ping test) — this remains open, decoupled from the vision-CNN pivot above, and is a reasonable place to make progress: write a digital testbench for `balance_controller()` first (AGENTS.md's verification sequence step 1), since it needs no physical hardware, no optical-platform spec, and no vision-model checkpoint.
- The `CodeV_Local` MCP tool (`generate_verilog`) exists for Verilog/HLS translation, but is currently **not to be invoked** — this machine does not have enough local compute to run the CodeV-DS-6.7B model. Leave the infrastructure in place for when that capacity exists; do not attempt Verilog translation via this tool until told the compute constraint is resolved.
- FPGA verification sequence:
  1. Write and check testbenches digitally first, before flashing any physical hardware.
  2. Verify camera input (VGA display/monitor sanity check, then route frame via UART/UDP to laptop for single-frame analysis).
  3. Verify motor control (90° rotation test, return to zero).
  4. Verify microphone input at 16kHz (store in RAM, send via UART, run inference on CPU).
  5. Once the optical platform's spec exists: design and verify the digital↔optical bridge against it.

### Phase 6 — Multimodal / VLA Comparative Study (ACTIVE, REFRAMED 2026-08-13 — owned by `.agents/agent_ml_multimodal.md`)

- Independent of Phases 1-5: does not draw against the FPGA resource budget (FPGA isn't running any model inference for this study — see Phase 5). Runs in parallel with the Vision/Audio/FPGA tracks.
- **This is now a 3-arm comparison, not a 2-arm one:**
  1. **Expert pipeline** — vision + audio + control, chained via Fusion. Which vision model backs this (Shared Backbone CNN vs. the older YOLO/ResNet lineage) is an **open question** — see the "Open question" callout in §"ML Component Sizes" above; do not assume an answer. Should deploy to the **Jetson Nano** in addition to the laptop, to remove hardware-platform as a confound against arm 2 — agreed direction (2026-08-13), not yet implemented.
  2. **Large (billions-of-param) Qwen-derived model** — adapted from the lab partner's model, deployed standalone on the **Jetson Orin Nano, 8GB** (own camera, own inference, own control loop — not relayed through the FPGA; see rationale in `host_software/ml_jetson_vla/docs/ARCHITECTURE.md`). This is the module this directory is for. The in-house `RT1LiteVLA` (`host_software/ml_multimodal/`) is a secondary/optional baseline now, not this arm.
  3. **Optical/photonic computing platform** — runs a model compiled from a similarly-sized pretrained VLA/LLM. FPGA bridges digital↔optical signals for this arm only (see Phase 5) — **blocked** on the platform's interface spec.
- `host_software/ml_multimodal/`'s existing scaffold (dataset synthesis → Stage 1 BC → Stage 2 RL → evaluation) still applies to `RT1LiteVLA` as the secondary baseline; several pieces are stubbed/mocked (dummy images, mock marker coords) — audit before trusting comparative numbers from it.
- `host_software/evaluations/evaluate_system_control.py` is the shared comparison tool across all three arms (`--runs label=csv label=csv ...`) — fixed 2026-08-13 to normalize timestamp-column differences across the expert/VLA CSV schemas and to match the documented metric definitions.
- Related external context: a lab partner is building a comparable Qwen-based model for the same task, which arm 2 adapts from — see "External / Comparative Context" above.

---

## Agent Capabilities
- **Verilog Translation (NOT CURRENTLY AVAILABLE):** An MCP server (`VRI_2026_AI_Router`) exposing `generate_verilog` (powered by the local CodeV-DS-6.7B LLM running via Ollama) is wired up in `.agents/codev_mcp.py` and registered project-wide via `.mcp.json`, but this machine lacks the local compute to actually run it. Treat this as infrastructure held in reserve for Phase 5 (FPGA Port), not a tool to call today — do not invoke `generate_verilog` until the compute constraint is resolved.
- **Large-File / Log Review (`ask_gemini_context`):** Same MCP server, for condensing large files/logs before writing implementation logic (see `.agents/agent_main.md` §2). Requires a `GEMINI_API_KEY` in `.env` at the repo root — get one free at [aistudio.google.com/apikey](https://aistudio.google.com/apikey), no paid plan required.
- **LaTeX Sheets:** Platform templates are generated procedurally in LaTeX. When adding new sheets, derive all coordinates from the physical platform dimensions (187.5 × 142.0 mm) and update `ground_truth_manifest.json` to match.
