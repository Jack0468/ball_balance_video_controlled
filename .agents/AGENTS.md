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
- **Microphone (Planned):** INMP441 digital I2S mic (3.3V, cheap, 16kHz).

---

## Current State of the Project

| Module | Status | Notes |
|--------|--------|-------|
| Vision (Ball) | ✅ Working | Premier pipeline: ArUco homography + 2D CNN (`cnn_2d_tracker_0730_v3`) + MLP corrector |
| Vision (Markers) | 🔧 In Progress | **Shared Backbone CNN planned** — see `implementation_plan_shared_backbone_cnn.md` |
| Audio | ⚠️ Needs Debug | Produces incorrect outputs during concurrent robot operation; matched filter suspected |
| Fusion | ⬜ Not Started | Requires working vision (markers) + verified audio |
| FPGA Port | ⬜ Not Started | Awaiting compact CNN model (~70K params); CodeV MCP available for Verilog translation |
| Hardware (Tripod) | ⬜ Not Started | Camera/mic mount design needed |
| Hardware (Power) | ⬜ Not Started | 24V 6A supply, FPGA power, PCB consolidation TBD |

---

## ML Component Sizes (Reference Baselines)

| Component | Model | Params | Status |
|-----------|-------|--------|--------|
| Ball Tracker | `cnn_2d_tracker_0730_v3` (BasicCNN) | ~340K | ✅ Working — too large for BRAM alone |
| MLP Corrector | `mlp_corrector_*` | ~70K | ✅ Working — being eliminated in new architecture |
| **Shared Backbone CNN** | `train_cnn_2d_tracker_marker` | **~70K total** | 🔧 Target architecture |
| Audio Classifier | Conv1D × 3 + Dense | ~13.5K | ⚠️ Needs debugging |
| Control Net | Small MLP (9→32→32→3) | ~1.5K | ✅ Working |
| VLA (RT1LiteVLA) | ResNet18 + Transformer | ~12M+ | ⬜ Not for FPGA — host PC only |

> [!IMPORTANT]
> The original YOLOv8-nano baseline is ~3.2M params. It is **far too large** for FPGA deployment and must never be used as a new model target. Use the ~70K Shared Backbone CNN instead.

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

## Data & Training Rules

- **Data collection** uses the existing `firmware/BallBalancingBot` (random ball movement) with `host_software/data_collection/collect_webcam_data.py`. Do NOT propose new firmware for this.
- **4 data sessions must be recorded**, one per platform configuration:
  1. Blank platform (`aruco.tex`) — used for synthetic marker compositing
  2. `aruco_markers_01.tex`
  3. `aruco_markers_02.tex`
  4. `aruco_markers_03.tex`
- **Training data split:** 60% synthetic composites (blank session) + 40% real printed sheets (01–03).
- **Evaluation is ONLY run on new, unseen sheets** (e.g., `aruco_markers_04.tex`). Never evaluate on training sheets.
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
- `models/`: Saved `.onnx` and `.pt` model weights. Organised by model name/version subdirectory.

### 2. Hardware & Firmware
- `hardware/platform_templates/`: Procedural LaTeX sheets and `ground_truth_manifest.json` (source of truth for marker physical coordinates).
- `firmware/BallBalancingBot/`: Single source of truth for STM32 telemetry, motor control, and PID. Do not add new firmware sketches for data collection.
- `host_software/data_collection/`: Python scripts for webcam and STM32 interaction. Key files: `collect_webcam_data.py`, `sync_webcam_telemetry.py`.

### 3. Documentation
- `docs/plans/`: Architecture and implementation plans. **Always read the relevant plan before writing code.**
  - `implementation_plan_shared_backbone_cnn.md` — Active implementation plan for the vision system.
  - `ml_system_parameter_budget.md` — FPGA resource budget. Consult this before designing any model.

### 4. Deprecated / Off-limits Directories
- `host_software/experimental_variants/`: Legacy experimental scripts. Read for reference only; do not add new files here.
- `host_software/new_vla_files/`, `host_software/ml_audio/`, `host_software/ml_multimodal/`: Inactive modules. Do not modify.

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
- Generate evaluation metrics report for CPU implementation before FPGA port.

### Phase 4 — Hardware (Tripod & Power)
- Design 3D-printed camera + microphone mount (tripod). Consider angle calibration hard-stops.
- Route cables for signal integrity (short as possible). FPGA should sit at base for weight stability.
- Power supply design: 24V 6A, FPGA power via voltage regulator from rail.
- Consider consolidated PCB: motor drivers, logic, audio circuit, power supply.

### Phase 5 — FPGA Port
- Target: ZedBoard (Xilinx XC7Z020-1CSG484CES).
- Treat ML inference as a "black box" — define hard-coded I/O contracts first.
- Use the `CodeV_Local` MCP tool for all Verilog/HLS translation tasks.
- FPGA verification sequence:
  1. Verify camera input (route frame via UART/UDP to laptop for single-frame analysis).
  2. Verify motor control (90° rotation test, return to zero).
  3. Verify microphone input at 16kHz (store in RAM, send via UART, run inference on CPU).
  4. Combine model weights into a single BRAM-resident block.
  5. Write and check testbenches digitally before flashing physical hardware.

---

## Agent Capabilities
- **Verilog Translation:** An MCP server (`CodeV_Local`) is available. When translating Python algorithms to FPGA Verilog, you MUST use the `generate_verilog` MCP tool (powered by the local CodeV-DS-6.7B LLM running via Ollama).
- **LaTeX Sheets:** Platform templates are generated procedurally in LaTeX. When adding new sheets, derive all coordinates from the physical platform dimensions (187.5 × 142.0 mm) and update `ground_truth_manifest.json` to match.
