# Large VLA Research Spike (Track 4) — Findings, 2026-08-18 (updated same day)

**Status: research findings, not an architecture decision.** Per the Jetson port plan,
this track is deliberately scoped as a spike, not a build. Nothing here is committed —
it's the input to a follow-up planning pass once these findings are checked against real
availability/licensing.

## Why AGENTS.md's existing 3-candidate survey isn't enough

`.agents/AGENTS.md`'s "Deployment feasibility survey" section already covers three
models on the confirmed Jetson AGX Orin 64GB (p3730):

| Candidate | Published rate on Jetson AGX Orin |
|---|---|
| Octo-small (~93M) | ~5-8Hz |
| Qwen2.5-VL-3B (INT4) | few tokens/sec (as a discrete planner, not per-frame control) |
| OpenVLA (7B) | ~2Hz |

None reach the ≥30Hz this project's control loop needs. That's expected, not a sign any
of these were mis-surveyed — no published single-rate monolithic VLA in current work
reaches real-time ball-balancing-class control rates. The fix isn't a faster monolith.

## The decoupled/dual-rate answer

The existing expert pipeline already solves this shape of problem: audio commands arrive
discretely/slowly (a "target" gets set), while a tiny 1.5K-param control net runs the
tight loop at full rate. The same split generalizes to a large-VLA arm — confirmed by
this session's research (web search, 2026-08-18) to be the actual industry-standard
answer, not a novel risk:

- **NVIDIA Isaac GR00T N1** (announced GTC 2026) ships this exact pattern in production: a
  "System 2" vision-language module for scene/language grounding, paired with a fast
  "System 1" diffusion-transformer action module for real-time motor output.
  Jetson-deployable via TensorRT, open weights on GitHub (`NVIDIA/Isaac-GR00T`). Almost
  certainly oversized/humanoid-manipulation-oriented for a 3-DOF ball-balancer — worth
  reading as the reference architecture even if the literal model is too heavy to run or
  retrain here.
- **LiteVLA-H** — 256M params, explicitly dual-rate *on Jetson AGX Orin (our exact
  hardware)*: ~19.74Hz outer-loop action emission, ~6.08-6.67Hz sentence-level semantic
  perception. Closest direct match found to the shape being proposed. Release status
  (code/weights) unconfirmed — treat as a paper-only reference until checked.
- **VOTE** (Vision-Language-Action Optimization) — reports 46Hz throughput on Jetson Orin,
  a 38.6x speedup over OpenVLA. **Correction from the first pass of this spike**: per a
  closer read of the paper, VOTE's *inference time* is actually higher than FASTER's
  approach — it's only faster than the (much slower) baselines VOTE compares itself
  against. The 46Hz headline shouldn't be read as "the fastest option found." Still needs
  verification against our exact AGX Orin 64GB variant and a task comparable in complexity
  to ball balancing, not a cherry-picked simple one.
- **FASTER** (real-time flow VLAs) — the key theoretical framing for this whole track, not
  just a technique. Its core finding: reaction time is **not** a fixed constant set by
  inference latency, but a random variable (modeled as uniform), because external events
  land at arbitrary phase relative to the controller's own inference cycle. Existing
  asynchronous-inference methods are "inherently limited" on their own — real
  responsiveness needs *both* lower perception-execution latency *and* a higher
  inference-execution cycle frequency together, not just one. This is why "just run the
  outer VLA a bit faster" wouldn't fix the real problem even if it were possible: the
  mismatch between when the world changes and when the model reacts is what matters, not
  raw Hz alone. FASTER targets flow-matching VLAs specifically — the same model family as
  the leading candidate below.
- **Jetson-PI** (`github.com/PKU-SEC-Lab/Jetson-PI`, checked 2026-08-18) — **the strongest
  candidate found, because it's the only one confirmed to have runnable code and
  downloadable weights**, not just a paper. Built on **Physical Intelligence's π0.5 /
  OpenPI** model family (Apache 2.0, plus a separate Gemma license for included
  components). Architecture: a lightweight "future correction" world model predicts where
  the environment *will be* by the time an action actually executes, and the action expert
  predicts from that predicted future state rather than the stale observed one — a
  concrete implementation of the fix FASTER's finding calls for (collaborative improvement
  of latency *and* cycle frequency), not just "also dual-rate." Trained/evaluated on LIBERO
  (sim) in the repo as shipped — no confirmed real-Jetson latency numbers publicly
  available; that still needs to be measured on our own hardware, not assumed from the
  paper. Checkpoints on Hugging Face / ModelScope; a separate `Jetson-PI-Edge` repo holds
  the accelerated inference engine.

## Recommended shape (not yet a final pick)

Keeps `ARCHITECTURE.md`'s locked framing that arm 2/3 are standalone (own camera, own
inference, own control loop) — i.e. does **not** reuse arm 1's vision CNN or control net,
which would confound "which architecture handles the task better":

- **Outer tier (slow, ~5-20Hz):** vision-language grounding of spoken commands + scene
  understanding into a high-level target. Real language understanding (not the closed
  5-word audio vocabulary the expert pipeline and `RT1LiteVLA` both use). Candidates:
  Qwen2.5-VL-3B (INT4), LiteVLA-H's perception tier, or π0.5's own VLM backbone if
  Jetson-PI is adopted wholesale.
- **Inner tier (fast, ≥30Hz):** given Jetson-PI/π0.5 has real, fine-tunable weights,
  **prefer fine-tuning it on our own data over designing a new action-expert architecture
  from scratch** — lower risk, and it already ships the foresight-alignment mechanism
  FASTER's finding calls for, which we would otherwise have to build ourselves. Fall back
  to a from-scratch compact action-expert only if Jetson-PI/π0.5 doesn't fit our latency
  budget once actually measured on our hardware.

## Training-data strategy for the inner tier (resolved 2026-08-18)

Real concern raised this session: a script that drives the platform directly to each
marker location gives exact, clean labels, but isn't robust — it only covers "go straight
to a marker" motions, with no recovery-from-disturbance, no off-marker intermediate
states, none of the diversity a real closed-loop controller actually encounters.

**Resolution: mirror this project's own existing pattern instead of inventing a new one.**
`ml_multimodal/training/train_vla.py`'s Stage 1 BC training already does exactly this — it
clones the expert pipeline's *live, closed-loop* behavior (including its PID jitter)
rather than scripted trajectories, precisely because that's what captures real operating
diversity. Recommendation: once Track 1 is running on the Jetson, log its real closed-loop
operation (camera frame, command/target, control-net input/output, actual ball trajectory)
as the **primary training set** for this inner tier — the same expert-pipeline-as-teacher
pattern, just retargeted from `RT1LiteVLA` to whichever inner-tier model this track lands
on. Scripted straight-to-marker runs are still useful, but only as a **minor supplement
for guaranteed endpoint coverage**, mirroring how the vision pipeline already treats
synthetic data (`AGENTS.md`'s 60/40 synthetic/real split, synthetic deliberately biased
toward gaps missing from real data) — a supplement to real closed-loop data, not a
replacement for it. Sets up a natural DAgger-style follow-up later (collect corrections
from the new policy's own live runs once a first version exists) if pure BC isn't enough
on its own — a future refinement, not part of this spike.

## Next steps (not started)

1. Prototype fine-tuning Jetson-PI/π0.5 specifically — it's the one candidate with real,
   checked-out code and weights, so it's the concrete next action rather than a generic
   "check availability" pass. LiteVLA-H and GR00T N1 remain architectural references, not
   confirmed buildable targets yet.
2. Measure Jetson-PI's actual latency on our own Jetson AGX Orin 64GB — its own repo's
   numbers are LIBERO-sim only, not from real Jetson hardware.
3. Start logging Track 1's live closed-loop runs (once bench-validated on real hardware)
   as the inner tier's primary training set, per the data-strategy resolution above — can
   start independent of which inner-tier model is finally chosen.
4. Write a follow-up recommendation doc once (1)-(3) have real numbers, not published
   benchmarks from other tasks/hardware.

`RT1LiteVLA`'s existing BC/RL scaffold in `ml_multimodal/` remains the secondary baseline
regardless of what this track lands on, per the project's existing framing.

---
Sources (web research, 2026-08-18):
- [Jetson-PI: Towards Onboard Real-Time Robot Control via Foresight-Aligned Asynchronous Inference](https://arxiv.org/html/2607.12659v3)
- [GitHub — PKU-SEC-Lab/Jetson-PI](https://github.com/PKU-SEC-Lab/Jetson-PI)
- [LiteVLA-H: Dual-Rate Vision-Language-Action Inference for Onboard Aerial Guidance and Semantic Perception](https://arxiv.org/html/2605.00884)
- [FASTER: Rethinking Real-Time Flow VLAs](https://arxiv.org/pdf/2603.19199)
- [VOTE: Vision-Language-Action Optimization (OpenReview)](https://openreview.net/pdf?id=jAWveMzE1p)
- [GitHub — NVIDIA/Isaac-GR00T](https://github.com/Nvidia/Isaac-GR00T)
- [NVIDIA Isaac GR00T N1 announcement](https://nvidianews.nvidia.com/news/nvidia-isaac-gr00t-n1-open-humanoid-robot-foundation-model-simulation-frameworks)
- [GR00T N1 paper (arXiv 2503.14734)](https://arxiv.org/abs/2503.14734)
