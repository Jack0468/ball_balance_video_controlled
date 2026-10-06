# VLA docs index (Track 4 / Arm 2)

Start here. Each entry gives the doc's job and its status. Status is what matters: some docs were superseded
by later decisions, and the newest section of each doc is authoritative.

## Start here: current decisions and status

| Doc | What it is | Status |
|---|---|---|
| `MULTI_HEAD_ARCHITECTURE_SPEC.md` | Decisions (2026-10-06), risk register, and the Qwen multi-head design | Top section is current. The Qwen backbone decision is under review, since the custom heads are being weighed against SmolVLA/ACT |
| `LARGE_VLA_RESEARCH_SPIKE.md` | Research on existing action models, dual-rate and action chunking; the SmolVLA recommendation (2026-09-15) | Current research. Its Jetson-PI numbers are ruled out |
| `ARM2_MINIMAL_BASELINE_MULTI_CANDIDATE.md` | Measured results for Qwen, PaliGemma2, InternVL2.5, Moondream2 (§8 to §13) | Current. The executive summary at the top is the fastest read |
| `ARM2_MINIMAL_BASELINE_SCOPE.md` | Scope and comparison axis for Arm 2 (effort/generality, not peak performance) | Still the framing doc; the specialization track is gated on its baseline sweep |
| `ACTION_CHUNK_CONTROL_SUBSYSTEM_BOOTSTRAP.md` | Offline check of action-chunk open-loop execution on the 10 Track 4 sessions | Current result set |

## How-to runbooks

| Doc | What it covers |
|---|---|
| `JETSON_ARM2_SWEEP_LOCAL.md` | Running the Arm 2 sweep on the Jetson: images, variables, ordered commands |
| `JETSON_ENV_SETUP.md` | Track 1 (expert pipeline) environment on the Jetson |
| `JETSON_FLASH_PROCEDURE.md` | Flashing the Jetson |
| `JETSON_VNC_DEBUG_ACCESS.md` | Start/stop VNC for debugging the Jetson |

## Background and planning (older, read for context)

| Doc | What it covers | Status |
|---|---|---|
| `ARCHITECTURE.md` | Overall ml_jetson_vla architecture and status | Background |
| `BOOTSTRAP_MODEL_COMPARISON_PLAN.md` | Metric B: comparing frozen backbones on one future point | Background; does not answer chunk questions |
| `MULTI_HEAD_OUTPUT_DESIGN.md` | Earlier multi-head output plan (2026-09-15) | Superseded by `MULTI_HEAD_ARCHITECTURE_SPEC.md` |
| `VISION_CALIBRATION_PROPOSAL.md` | Per-session vision calibration proposal | Proposal |
| `dependency_on_jetson_notes.md` | Notes on Jetson dependencies | Notes |

## Data pipeline (in `host_software/ml_multimodal/docs/`)

| Doc | What it covers |
|---|---|
| `DATA_PIPELINE.md` | VLA data collection and training pipeline |

## Repo-level docs (in `docs/`)

| Doc | What it covers |
|---|---|
| `../../../docs/PROJECT_LOGBOOK.md` | Append-only history and the reasoning behind decisions. Read it for "why" |
| `../../../docs/EVALUATION_STRATEGY.md` | The four control metrics (steady-state error, settling time, control effort, task success) |
| `../../../docs/DATA_STORAGE.md` | Data and model storage policy |
| `../../../docs/CAMERA_HARDWARE.md` | Jetson USB webcam setup |
| `../../../docs/HARDWARE_AND_SOFTWARE_PREREQUISITES.md` | Prerequisites for the whole project |
| `../../../docs/plans/` | Vision and FPGA plans (shared backbone CNN, marker CNN, parameter budget, ArUco confusion root cause) |
| `../../../docs/archive/` | Retired planning docs. Do not use as current |

## Conventions

- Results go into the doc that owns the topic, as a dated section. They never go into `host_software/data/`, which is gitignored.
- The newest dated section of a doc overrides older sections, and the executive summary overrides the detail.
- Anything marked "superseded" points to its replacement; don't implement from it.
