# ml_jetson_vla — Architecture & Status

Captures the system-level comparison this module exists for. Read this before writing code
here — the framing has changed several times across one long conversation (2026-08-13 to
2026-08-14) and it's easy to build against a superseded version of it. Each section below is
dated; trust the newest date on a given point.

## THIS IS THE PROJECT'S MAJOR GOAL RIGHT NOW (2026-08-14)

Two hardware decisions are locked in, and this directory's job is now explicitly the
project's top priority — ahead of the FPGA/photonic track, which remains real and documented
but is not the current focus:

- **Camera: USB webcam directly into the Jetson.** No laptop relay, no FPGA/OV7670 path.
  Reuses existing `cv2.VideoCapture` code verbatim. Laptop→network→Jetson streaming was
  rejected (network jitter, especially over WiFi, risks corrupting the RL control net's
  timing-sensitive velocity filter). OV7670-direct-to-Jetson was rejected (same
  OS-scheduling-jitter problem as driving steppers from Jetson — no native USB/CSI interface,
  would need timing-critical parallel-bus bit-banging in Linux userspace).
- **Motor actuation: STM32, via `SerialCoords.cpp`, for now.** FPGA Option B
  (`stepper_motor_controller.v` ported to a Zynq AXI-Lite IP) is deferred, not abandoned —
  only pick it up if a real project constraint later forces it.

**The goal:** evaluate small, medium, and large model classes — all running on the Jetson —
and compare them on the project's standard four metrics.

## The comparison being built (updated 2026-08-14)

Three parallel arms perform the same ball-balancing/target-reaching task, compared on the
project's standard four metrics (`docs/EVALUATION_STRATEGY.md`, computed by
`host_software/evaluations/evaluate_system_control.py`). **All three arms share the Jetson
Nano where physically possible**, specifically to control for hardware platform as a
confound — arms 1 and 2 both run on it; arm 3 can't by definition (different compute
substrate is the point of that arm).

1. **Expert system** — CNN vision + `ml_audio` (command classifier) + the RL-trained control
   net, chained via Fusion. In progress; the vision model is training as of this writing.
   **All three components should deploy to the Jetson AGX Orin** (not just the laptop) — agreed
   2026-08-13, not yet implemented — to remove hardware-platform as a confound against arm 2.
   Physical actuation still needs an external hop regardless of where the "brain" computing
   targets runs: the Jetson can't reliably generate STEP/DIR pulses itself (Linux scheduling
   jitter, confirmed by reading the actual `AccelStepper`-based firmware — see
   `.agents/agent_fpga.md`'s "Option B" note). Either the STM32 (`SerialCoords.cpp`, already
   built, accepts exactly this kind of external-host input over serial) or the FPGA's
   `stepper_motor_controller.v` (Option B, under investigation, not yet built) executes them.

   **Control net I/O contract (read from `RLControl.cpp` directly, 2026-08-14):** a
   9→32→32→3 tanh MLP (~1.5K params). Input: 9-dim raw-unit vector
   `[ball_x, ball_y, x_error, y_error, filtered_vel_x, filtered_vel_y, actual_step_A,
   actual_step_B, actual_step_C]` — velocity is an EMA-filtered finite difference carried as
   state across calls, and `actual_step_*` is the motors' *real, lag-affected* position, not
   the last commanded target (the network was trained on lagged state specifically). Output:
   3 values in `[-1,1]`, scaled by 98 and rounded → **raw step-space targets directly** — no
   `InverseKinematics.cpp` step in this path at all. Porting this to the Jetson is
   computationally trivial (microseconds anywhere) but requires faithfully replicating the
   velocity filter and the closed-loop actual-position feedback — get either wrong and the
   policy runs outside its training distribution, since it's learned, not hand-tuned.

   **Vision sub-choice is a taxonomy, not a single pick (2026-08-14):** a **"small model"
   class** (Shared Backbone CNN, ~64-91K params depending on decoder choice, FPGA-target
   design, training now) and a **"medium model" class** (the YOLO/ResNet lineage
   `run_eval_expert.py` actually runs today — `yolov8_platform_pose_markers_iphone_v1` +
   `mlp_corrector_iphone_v1`, plus `cnn_2d_tracker_0730_v3`/BasicCNN, YOLOv8-nano, ResNet18/50
   variants). Both aim at the same vision task on different-but-similar data — see
   `AGENTS.md`'s "Arm 1 has two vision model classes" callout under ML Component Sizes. The
   lurking variable is training-data recency (only the small class has seen Dataset 8/9, the
   corrected/complete data), not architecture size. **Neither is decided; both are live
   candidates**, pending the small class's evaluation once its current training run finishes.

2. **Jetson AGX Orin, large pretrained model** — an adapted version of the lab partner's
   Qwen-based model (NOT the small in-house `RT1LiteVLA` in `ml_multimodal/`), running
   **standalone** on the Jetson AGX Orin 64GB Developer Kit model: p3730: own camera, own inference, own control loop.
   This is what this directory (`ml_jetson_vla/`) is for.

3. **Photonic computing platform — confirmed 2026-08-14 as a genuine end-to-end execution
   target**, not a partial matrix-multiply accelerator sitting behind other digital compute.
   Runs **the same class of large VLA model as arm 2** — an earlier framing considered
   same-day (that it would instead run photonic-compiled versions of *our own* small/medium
   expert models) was superseded; don't resurrect it without checking first. FPGA's role, as
   currently described: (1) sample the camera (`fpga/camera_i2c/`, already built), (2)
   electro-optically encode the signal for the photonic network's input, (3) read the
   photonic network's output back, (4) convert that into motor angles and/or STEP/DIR. Since
   the model is VLA-scale, its output is expected to be `theta_a/b/c` angles (matching the
   `ml_multimodal`/here training schema), meaning step 4 needs `angle_to_steps()`
   (`firmware/stm32_ml_control_and_vision/BallBalancingBot/MotorControl.h`), unlike arm 1's
   control net which needs no conversion. **Still blocked**: no interface spec exists for
   steps 2/3 (modulator driver? photodetector ADC? timing/protocol?) — do not design real
   Verilog against a guessed protocol. Carrying camera-frame-scale data across this bridge
   (step 2) is comparable in scale/risk to the still-unresolved camera→UDP video streaming
   problem — don't assume it's an easier problem just because it's a different signal domain.
   This arm is `fpga/`'s scope (`.agents/agent_fpga.md`), not this directory's.

### What changed from the original plan

The FPGA was originally being developed to *run* the vision CNN on-chip (see
`docs/plans/ml_system_parameter_budget.md` §5, `fpga/hls4ml_custom_layers/` — GELU LUT layer
built and verified, `Upsample+Conv2d` decoder architecture reconstructed and param-matched).
That work is **paused, not deleted** — it was real, verified engineering and may become
relevant again if the architecture changes, but it is not the active integration target now
that arm 3 is confirmed to run the large VLA model rather than our own small/medium models.

### Why "standalone" for the Jetson arm, not FPGA-relayed sensor data

Two options were raised: (a) Jetson gets sensor data relayed through the FPGA, or (b) a fully
standalone Jetson system with its own camera. Defaulted to **(b) standalone** because:
- The FPGA's camera→UDP video pipeline has never worked end-to-end
  (`docs/udp_research_guide.md` still has six untested failure hypotheses for it) — making
  the Jetson arm depend on that pipeline would block arm 2's progress on arm 3's unresolved
  hardware bring-up, for no architectural reason (nothing about a general-purpose VLA
  requires FPGA-mediated sensor input).
- Standalone keeps this arm's development, and eventually its evaluation runs, fully
  decoupled from FPGA/optical-platform readiness.

**This is a default, not a locked decision** — revisit if it turns out the optical arm's
camera rig needs to be shared with the Jetson arm for a fair comparison (identical sensor
conditions across arms 2 and 3), or if there's only one physical camera/platform rig
available for testing.

## What's confirmed vs. still open

| Item | Status |
|---|---|
| Jetson hardware | Confirmed: Jetson AGX Orin 64GB Developer Kit model: p3730, Ampere/Tensor Cores |
| Realistic model ceiling on this hardware | ~4B params, INT4-quantized, via MLC-LLM/TensorRT-LLM (NVIDIA's own Jetson AI Lab benchmarks target this class) — see `.agents/AGENTS.md`'s "External / Comparative Context" section |
| Jetson arm's model | The lab partner's Qwen-based model, adapted — **not yet in our possession**. Model variant/checkpoint/access path unconfirmed. |
| Jetson arm's sensor input | Standalone (own camera) — see rationale above, default not locked |
| Arm 1's vision model | Open — small class (Shared Backbone CNN) vs. medium class (YOLO/ResNet lineage), both live candidates, decision pending the small class's evaluation |
| Arm 3's model identity | Confirmed 2026-08-14: same class of large VLA model as arm 2, on a confirmed end-to-end photonic execution platform |
| Optical computing platform's electrical/protocol interface | **Unknown / not yet specified.** No datasheet, no confirmed device, no signal-level spec. Do not design real Verilog against a guessed protocol. |
| FPGA-side motor actuation ("Option B") | Under investigation, not yet built. `fpga/main_controller/stepper_motor_controller.v` exists (Opal Kelly-era but its core logic has zero Opal Kelly dependencies) and is a real candidate to port — see `.agents/agent_fpga.md` Domain 4 for the concrete porting plan and the precedent already in this repo (`ov7670_axi_stream.v`'s IP packaging). |
| Physical test data (any arm) | **None available yet.** Everything in this directory must be built as ready-to-run scaffolding against a real model/checkpoint/dataset, not asserted to work — mirrors the same situation `ml_multimodal/` was already in. |

## Module layout

- `core/policy_interface.py` — **built 2026-08-18.** Model-agnostic policy wrapper
  (`Policy` protocol, `act(image, instruction, state) -> PolicyCommand`), shared shape with
  `ml_multimodal`'s equivalent so both arms' eval runs can be driven by the same harness
  pattern (note: `ml_multimodal/` does not actually have a matching wrapper yet — its
  evaluators call `RT1LiteVLA.forward()` directly — so this defines the shape going
  forward rather than mirroring pre-existing code). `PolicyCommand.step_targets` is the
  Phase-B hook (None today; populated once control-net-on-Jetson exists).
- `deployment/` — quantization/export pipeline (INT4, MLC-LLM or TensorRT-LLM) for
  whichever model lands here. Currently a scaffold — there is no model to point it at yet.
- `runtime/run_jetson_standalone.py` — **built 2026-08-18 (Track 1, Phase A only).**
  Currently implements the *small-class expert pipeline* arm (vision+audio, CPU-only ONNX),
  not arm 2 — that's still blocked on the lab partner's model. Emits console status, not
  yet a CSV (see `runtime/telemetry_logger.py` below for why). `JetsonExpertPolicy` inside
  this file is the first concrete `Policy` implementation.
- `runtime/telemetry_logger.py` — **built 2026-08-18.** Jetson-side parser for the STM32
  telemetry proposal in `stm32_interface/` (not yet merged into firmware — see below).
  Once merged, writes exactly `evaluate_system_control.py`'s `REQUIRED_COLUMNS` schema
  (`host_timestamp_ms`, `target_x/y`, `touch_x/y`, `theta_a/b/c`) so any arm's run plugs
  into the existing comparison tool with zero changes to it.
- `stm32_interface/` — **new 2026-08-18, proposal only, not merged.** `firmware/` is
  outside this directory's write boundary, so `Telemetry.cpp/.h` here is a reference
  implementation staged for the Electrical Engineer to review, not live code. Exists
  because `BallBalancingBot.ino` (current source-of-truth firmware) turned out to have no
  telemetry-back at all under its live `SerialCoords.cpp` path — a gap independent of the
  Jetson port itself, but blocking for "record accuracy so arms can be compared." See
  `stm32_interface/TELEMETRY_PROTOCOL.md`.
- No local `evaluations/` — reuses `host_software/evaluations/evaluate_system_control.py`
  directly (multi-run `--runs` mode) rather than duplicating metric logic per-module.

## Immediate blockers (not this directory's job to resolve)

1. Get the lab partner's Qwen model/weights/access path (arm 2) — **now has a parallel,
   unblocked path**: Track 4 (`docs/LARGE_VLA_RESEARCH_SPIKE.md`, 2026-08-18, revised
   2026-08-19) scopes an in-house large-VLA alternative instead of only waiting on this —
   dual-rate architecture, outer tier = Jetson-PI/π0.5 fine-tuned on our own data (its
   measured ~2.4Hz on Jetson Orin rules it out as the fast tier, so it's now scoped as the
   slow grounding tier instead), inner tier = a fast custom action-expert, architecture
   not yet designed. Concrete pipeline stages (data conversion to LeRobot format,
   LoRA/Colab fine-tuning, GGUF export via Jetson-PI-Edge) are spelled out in that doc.
2. Get the optical computing platform's real interface spec before any FPGA bridge Verilog
   is written (arm 3).
3. Arm 1's vision model choice (small vs. medium class) — pending the small class's
   evaluation once training finishes.
4. **New 2026-08-18:** `stm32_interface/`'s telemetry proposal needs Electrical Engineer
   review and merge into `firmware/` before any Jetson-hosted arm can produce a full
   four-metric-comparable CSV (Track 1 can still run live without it — it just can't be
   scored yet).
