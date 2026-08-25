# Large VLA Research Spike (Track 4) — Findings, 2026-08-18, revised 2026-08-19

**Status: moving from research spike into concrete pipeline planning** (2026-08-19) — the
architecture question (which model, which tier) is resolved enough to plan build stages
against; the inner-tier's own architecture is still open, see below.

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

## REVISED 2026-08-19 — Jetson-PI measured latency rules it out as the inner tier

`Jetson-PI-Edge`'s own published numbers (checked this session): **394.5ms/inference for
π0, 412.9ms for π0.5 on Jetson Orin — ~2.4Hz.** Not the ~19-46Hz range LiteVLA-H/VOTE
claim, and nowhere near the ≥30Hz the tight control loop needs. This contradicts the
earlier reading of the Jetson-PI *paper* (which frames its foresight-alignment mechanism
as enabling real-time control) — that framing is about effective responsiveness under
asynchronous prediction, not raw inference Hz, and the *edge-deployment* repo's own
measured number is what actually matters for our hardware. Decision (user, 2026-08-19):
don't spend time re-verifying whether the foresight trick makes 2.4Hz "feel" fast enough —
treat Jetson-PI as too slow for the inner tier and move on.

## Recommended shape (architecture resolved, inner tier still open)

Keeps `ARCHITECTURE.md`'s locked framing that arm 2/3 are standalone (own camera, own
inference, own control loop) — i.e. does **not** reuse arm 1's vision CNN or control net,
which would confound "which architecture handles the task better":

- **Outer tier (slow, ~2-20Hz) — Jetson-PI/π0.5, repurposed as the grounding tier.** Its
  measured ~2.4Hz is actually a fine fit here (comparable to Qwen2.5-VL-3B's "few
  tokens/sec" and OpenVLA's ~2Hz, both originally scoped as outer-tier candidates) — this
  doesn't waste the fine-tuning work, it just changes what role the result plays. Worth
  checking during implementation whether π0.5 can be used purely for vision-language
  grounding (scene + language → high-level target, replacing the closed 5-word audio
  vocabulary) while ignoring its own action-expert output, since a separate inner tier now
  owns actions — would simplify the fine-tuning objective from "imitate full VLA control"
  to something closer to grounding/classification. Not confirmed whether π0.5's
  architecture cleanly supports this split; verify, don't assume.
- **Inner tier (fast, ≥30Hz) — custom, trained on our own data.** Promoted from fallback to
  primary now that Jetson-PI has failed the latency bar. Architecture not yet designed —
  the least-specified part of this whole track, needs its own design pass once the
  outer-tier pipeline is underway (not blocking it).

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

## Concrete pipeline (2026-08-19)

1. **Data collection** — blocked on Track 1 running on real Jetson hardware. Logs camera
   frame, recognized command, ball trajectory, control-net input/output. Feeds both the
   outer tier's grounding fine-tune and, eventually, the inner tier's training set.
2. **Data conversion to LeRobot format** — new work, not yet built. Confirmed schema
   (LeRobotDataset v3.0, checked 2026-08-19): `meta/info.json` (feature shapes/dtypes,
   fps, path templates), `meta/episodes/...parquet` (episode metadata),
   `meta/tasks.parquet` (task definitions — note: some docs still reference an older
   `tasks.jsonl`; confirm which version `openpi`/Jetson-PI's training stack actually
   expects before committing to one), `meta/stats.json` (aggregated stats, feeds
   `compute_norm_stats.py`), `data/chunk-*/*.parquet` (one row per timestep:
   episode_index, frame_index, timestamp, state vector, action vector, `next.done`),
   `videos/<camera_key>/chunk-*/*.mp4`. **Build the converter using the official
   `LeRobotDataset.create()` Python API** (feature dict → `add_frame()`-style calls →
   `save_episode()` → `finalize()`), not by hand-writing the parquet/video files directly —
   confirmed this is the supported path via `lerobot`'s own tutorial. Exact per-frame
   write-call signature not yet confirmed against real code (only tutorial-level examples
   read so far) — check `lerobot`'s actual API reference before writing this converter for
   real, don't guess the call shape from the tutorial's higher-level `record_loop()`
   wrapper.
3. **Fine-tuning compute: Google Colab, A100 tier.** Jetson-PI's own 3-stage full-finetune
   launcher wants ≥48GB VRAM — over Colab's typical ~40GB A100 allocation. Plan: use LoRA
   fine-tuning instead (`gemma_2b_lora`/`gemma_300m_lora`-style configs exist upstream in
   `Physical-Intelligence/openpi`, reported to fit in ~22.5GB+). **Open verification item**:
   whether Jetson-PI's own training scripts support LoRA the same way upstream `openpi`
   does, or whether fine-tuning needs to go through `openpi` directly with Jetson-PI only
   consuming the result — check before committing Colab time to a run that might not fit.
4. **Outer-tier export for Jetson inference.** `Jetson-PI-Edge` (llama.cpp-based) needs
   GGUF — both the PI language/action model and a SigLIP vision encoder/projector.
   Conversion path: its own "Model Preparation" guide (not yet read in detail). Pre-converted
   base (non-fine-tuned) BF16 GGUF checkpoints exist on Hugging Face
   (`diantoudefengshan/Jetson-PI-GGUF`) — smoke-test the export/serving path against these
   before trying to convert our own fine-tuned weights.
5. **Inner tier — design pass, not started.** Deliberately deferred; informed by what data
   logging (step 1) actually produces, not blocking steps 2-4.
6. **Integration** — wrap both tiers behind `core/policy_interface.py`'s `Policy` protocol
   (same interface Track 1's `JetsonExpertPolicy` already implements).

Sequencing: steps 2-3 (converter tooling, LoRA config check) can start now, independent of
Track 1 hardware bring-up. Step 1 needs real hardware. Step 4 can be smoke-tested against
the pre-converted base checkpoint before any fine-tuning finishes.

`RT1LiteVLA`'s existing BC/RL scaffold in `ml_multimodal/` remains the secondary baseline
regardless of what this track lands on, per the project's existing framing.

---
Sources (web research, 2026-08-18 and 2026-08-19):
- [Jetson-PI: Towards Onboard Real-Time Robot Control via Foresight-Aligned Asynchronous Inference](https://arxiv.org/html/2607.12659v3)
- [GitHub — PKU-SEC-Lab/Jetson-PI](https://github.com/PKU-SEC-Lab/Jetson-PI)
- [GitHub — PKU-SEC-Lab/Jetson-PI-Edge](https://github.com/PKU-SEC-Lab/Jetson-PI-Edge)
- [LiteVLA-H: Dual-Rate Vision-Language-Action Inference for Onboard Aerial Guidance and Semantic Perception](https://arxiv.org/html/2605.00884)
- [FASTER: Rethinking Real-Time Flow VLAs](https://arxiv.org/pdf/2603.19199)
- [VOTE: Vision-Language-Action Optimization (OpenReview)](https://openreview.net/pdf?id=jAWveMzE1p)
- [GitHub — NVIDIA/Isaac-GR00T](https://github.com/Nvidia/Isaac-GR00T)
- [NVIDIA Isaac GR00T N1 announcement](https://nvidianews.nvidia.com/news/nvidia-isaac-gr00t-n1-open-humanoid-robot-foundation-model-simulation-frameworks)
- [GR00T N1 paper (arXiv 2503.14734)](https://arxiv.org/abs/2503.14734)
- [GitHub — Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)
- [LeRobot: Imitation Learning on Real-World Robots (record/train tutorial)](https://huggingface.co/docs/lerobot/il_robots)
- [LeRobotDataset v3.0](https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3)
