# ml_jetson_vla — Architecture & Status

Captures the system-level comparison this module exists for, as clarified 2026-08-13.
Read this before writing code here — the framing changed twice in one conversation and
it's easy to build against the wrong version of it.

## The comparison being built (2026-08-13, current understanding)

Three parallel arms perform the same ball-balancing/target-reaching task, compared on the
project's standard four metrics (`docs/EVALUATION_STRATEGY.md`, computed by
`host_software/evaluations/evaluate_system_control.py`):

1. **Expert system** — the existing modular pipeline: `ml_vision` + `ml_audio` (command
   classifier) + a PID/IK control law, chained via Fusion. In progress; the vision model is
   training as of this writing. **Which vision model backs this arm is an open, undecided
   question** (Shared Backbone CNN vs. the older YOLO/ResNet lineage — see `AGENTS.md`'s
   "Open question" callout under ML Component Sizes): the older models were never trained on
   Dataset 8/9 (the current, corrected, most complete data), so an accuracy comparison
   between them right now would conflate architecture with data-recency, not isolate either.
   Don't assume an answer. **Should also deploy to the Jetson** (not just the laptop) —
   agreed 2026-08-13, not yet implemented — to remove hardware-platform as a confound
   against arm 2 below (the expert models are tiny, ~91K vision + 13.5K audio + 1.5K
   control, so this is a runtime-target change, not new engineering).
2. **Jetson Nano, large pretrained model** — an adapted version of the lab partner's
   Qwen-based model (NOT the small in-house `RT1LiteVLA` in `ml_multimodal/`), running
   **standalone** on the Jetson Orin Nano (8GB): own camera, own inference, own control loop.
   This is what this directory (`ml_jetson_vla/`) is for.
3. **Optical/photonic computing platform** — runs a model "compiled from" a pretrained
   VLA/LLM with similar characteristics to arm 2's. The FPGA's role here is to bridge
   digital sensor/control signals to and from the optical domain — it does **not** run its
   own inference in this arm ("not at the same time" as bridging, per 2026-08-13
   discussion). This is `fpga/`'s scope, not this directory's.

### What changed from the previous plan

The FPGA was previously being developed to *run* the vision CNN on-chip (see
`docs/plans/ml_system_parameter_budget.md` §5, `fpga/hls4ml_custom_layers/` — GELU LUT
layer built and verified, `Upsample+Conv2d` decoder architecture reconstructed and
param-matched). That work is **paused, not deleted** — it was real, verified engineering
(the GELU LUT conversion is confirmed against hls4ml's actual converter) and may become
relevant again if the optical-platform plan changes. But it is not the active integration
target: the FPGA's confirmed near-term job is the optical-platform interface, not on-chip
CNN inference. See `docs/plans/ml_system_parameter_budget.md`'s status note (added
2026-08-13) and `fpga/docs/` for the interface side.

### Why "standalone" for the Jetson arm, not FPGA-relayed sensor data

The user raised two options: (a) Jetson gets sensor data relayed through the FPGA, or (b) a
fully standalone Jetson system with its own camera. Defaulted to **(b) standalone** because:
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
| Jetson hardware | Confirmed: Jetson Orin Nano, 8GB, Ampere/Tensor Cores (not the older Maxwell Jetson Nano) |
| Realistic model ceiling on this hardware | ~4B params, INT4-quantized, via MLC-LLM/TensorRT-LLM (NVIDIA's own Jetson AI Lab benchmarks target this class) — see `.agents/AGENTS.md`'s "External / Comparative Context" section |
| Jetson arm's model | The lab partner's Qwen-based model, adapted — **not yet in our possession**. Model variant/checkpoint/access path unconfirmed. |
| Jetson arm's sensor input | Standalone (own camera) — see rationale above, default not locked |
| Optical computing platform's electrical/protocol interface | **Unknown / not yet specified.** No datasheet, no confirmed device, no signal-level spec (modulator driver? photodetector ADC? timing?). Do not design real Verilog against a guessed protocol. |
| Physical test data (any arm) | **None available yet** (2026-08-13). Everything in this directory must be built as ready-to-run scaffolding against a real model/checkpoint/dataset, not asserted to work — mirrors the same situation `ml_multimodal/` was already in. |

## Module layout

- `core/policy_interface.py` — model-agnostic policy wrapper (`act(image, instruction,
  state) -> command`), shared shape with `ml_multimodal`'s equivalent so both arms' eval
  runs can be driven by the same harness pattern. Reusable regardless of which Qwen variant
  ends up deployed.
- `deployment/` — quantization/export pipeline (INT4, MLC-LLM or TensorRT-LLM) for
  whichever model lands here. Currently a scaffold — there is no model to point it at yet.
- `runtime/run_jetson_standalone.py` — the on-device standalone loop (capture → inference →
  command → motor control), structured to emit the same CSV schema
  `evaluate_system_control.py` already normalizes (`host_timestamp_ms`, `target_x/y`,
  `touch_x/y`, `theta_a/b/c`) so this arm plugs into the existing comparison tool with zero
  changes to it once real hardware/model are available.
- No local `evaluations/` — reuses `host_software/evaluations/evaluate_system_control.py`
  directly (multi-run `--runs` mode) rather than duplicating metric logic per-module.

## Immediate blockers (not this directory's job to resolve)

1. Get the lab partner's Qwen model/weights/access path.
2. Get the optical computing platform's real interface spec before any FPGA bridge Verilog
   is written.
