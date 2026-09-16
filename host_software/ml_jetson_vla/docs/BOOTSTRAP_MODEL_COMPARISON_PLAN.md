# Bootstrap Model Comparison Plan (Track 4, item 4) — 2026-09-15

**Status: design/methodology only. No comparison run, no additional checkpoints downloaded.**
This is the plan item 5's Colab notebook should implement. It does not do item 5's work, and
it does not do item 7's control-subsystem prototyping, even though Metric B below shares a
"predict the near future" shape with item 7.

## 0. Why this plan looks the way it does

Item 3 (`MULTI_HEAD_ARCHITECTURE_SPEC.md`) established that **no arm-2 candidate has a
trained action head today** — Qwen2.5-VL-3B is a plain VLM (`generate()` → text, full stop),
and even the one candidate that does ship a native action head (Jetson-PI/π0.5) has never been
fine-tuned on this project's data. A fair pre-commitment comparison therefore cannot measure
closed-loop control quality (steady-state error, settling time, control effort, task success
rate — `docs/EVALUATION_STRATEGY.md`) for any candidate yet; those four metrics stay the
comparison basis **once a real policy exists per arm**, not before. This plan proposes two
cheap, representation-level proxies instead, computable today against frozen (or near-zero-cost
fine-tuned) backbones, against the real Track 4 data already on disk.

Per `model-iteration-constraints` (loaded before writing this): §3 (isolate the variable you're
testing) and §4 (never rank from a small seed count) shaped both metrics below more than
anything else — see §2.4 and §3.

## 1. Candidate shortlist

Built from `LARGE_VLA_RESEARCH_SPIKE.md`'s existing survey — not re-surveyed from scratch.
Scoped to genuinely large-class models per this task's routing (SmolVLA and Octo-small are
excluded as primary candidates for the reason stated there: they don't represent this
project's billions-of-param "large VLA" class as defined in `ARCHITECTURE.md`/`CLAUDE.md`).

| Candidate | Params | Status | Verdict |
|---|---|---|---|
| **Qwen2.5-VL-3B-Instruct** | ~3.74B (computed, `MULTI_HEAD_ARCHITECTURE_SPEC.md` §1) | Already downloaded (`models/qwen2_5_vl_3b_instruct/`), already smoke-tested end-to-end (`deployment/qwen_vl_smoke_test.py`, `generate()` confirmed working) | **Primary candidate.** Under the ~4B ceiling. Confirmed operative choice for arm 2 as of today's session (superseding the 2026-09-15 SmolVLA framing in `LARGE_VLA_RESEARCH_SPIKE.md` — see `MULTI_HEAD_OUTPUT_DESIGN.md` §0's own flagged inconsistency). Plausibly the closest architectural proxy to the lab partner's actual "Qwen-derived" model, per that doc's hypothesis (unconfirmed). |
| **Jetson-PI / π0.5 (OpenPI)** | ~3.3B (PaliGemma-3B backbone + ~300M action expert) | Not downloaded in this repo yet. Real, runnable code + downloadable checkpoints confirmed to exist (Apache-2.0 + separate Gemma license) — the research spike's own words: "the only one confirmed to have runnable code and downloadable weights, not just a paper" | **Secondary candidate.** Under the ~4B ceiling. Ruled out earlier (2026-08-19) as the *real-time control* inner tier on measured 2.4Hz Jetson-Edge latency — but that verdict is about control-loop rate, not representation quality, and is irrelevant to this proxy comparison. Worth testing here specifically *because* it already ships a trained vision→action-expert path (PaliGemma + flow-matching action head), giving a genuine "does a purpose-built action architecture already beat a linear probe on a plain VLM's features" contrast point (see Metric B). |
| **NVIDIA Isaac GR00T N1** | Unconfirmed | Open weights on GitHub (`NVIDIA/Isaac-GR00T`), but the research spike's own assessment is "almost certainly oversized/humanoid-manipulation-oriented... worth reading as reference architecture even if too heavy to run or retrain here" | **Conditional, gated — not committed.** Do not download blind. Needs a scoped pre-check (confirm smallest available variant's param count against the ~4B ceiling, confirm license terms) before it enters the actual comparison. If no sub-~4B variant exists, drop it rather than force a comparison against an over-ceiling model. |
| OpenVLA-7B | 7B | Already surveyed (`AGENTS.md`'s original 3-candidate table, ~2Hz on Jetson AGX Orin) | **Excluded.** Exceeds the ~4B realistic ceiling by ~1.75x, and the existing survey already found it slower than the other two surveyed candidates on this exact hardware class. Not re-included here without a specific reason to revisit. |
| LiteVLA-H | 256M | Paper-only reference in the spike; release status of code/weights unconfirmed | **Excluded.** Sub-large-class by param count (same reasoning as excluding SmolVLA/Octo-small), and no confirmed downloadable checkpoint exists to test against regardless. |
| **SmolVLA** | 450M | Explicitly redirected away from as an arm-2 candidate this session | **Not a shortlist candidate.** Included below only as an optional, explicitly-scoped *harness-validation* step (§4), not as a competitor in the actual comparison — see justification there. |

**Net: 2 firm candidates (Qwen2.5-VL-3B, Jetson-PI/π0.5), 1 conditional (GR00T N1, gated on a
cheap pre-check), 1 harness-only non-candidate (SmolVLA).** This is a thin shortlist, not
padded for its own sake — it reflects how few of the models this project has already surveyed
are both genuinely large-class *and* confirmed to have real, downloadable weights today.

## 2. Proxy evaluation methodology

Two metrics, targeting the two roles a multi-head design (item 3) splits apart: language+vision
**grounding** (Head A's job) and near-future **state/action-relevant prediction** (Heads B/C's
job). Both are computable against a frozen backbone — no BC/RL training loop, no PPO, matching
this task's explicit scope.

### 2.1 Metric A — Zero-shot language-grounded target localization accuracy

Tests whether a candidate's off-the-shelf vision-language grounding is already useful for this
task, with **zero fine-tuning** (uses each candidate's stock `generate()`/inference API as-is —
for Qwen2.5-VL-3B this is the exact call shape `qwen_vl_smoke_test.py` already exercises).

- **Input:** one frame at time *t* where the session's real `audio_command` at *t* is one of
  the 4 color-target commands (`go_red`/`go_green`/`go_yellow`/`go_black` — the only commands
  with a well-defined target marker location; directional commands `forward/left/right/
  backward/hold/stop` are out of scope for this metric, see §2.4).
- **Prompt (starting point, needs real-checkpoint verification before use):** *"The robot must
  move the ball to the {color} marker. Point to the marker's location in this image."* Qwen2.5-VL
  is documented upstream to support grounded point/bbox output — **not yet confirmed against
  this local checkpoint/transformers version**, flag as an open item for item 5 to check before
  committing to a parser.
- **Ground truth:** the session's own recorded `target_x`/`target_y` (mm, `telemetry.csv`),
  converted to pixel space using `host_software/ml_vision/core/coordinate_math.py`'s
  `HomographyProjector`/`PixelToPhysicalMapper` (read-only reuse — `ml_vision` is not this
  agent's territory to modify, and reusing its already-built, calibrated mapping avoids
  duplicating homography logic per `feedback_reuse_existing_export_tooling`).
- **Scoring:** parse the candidate's output into a point (or bbox center); a frame is a **hit**
  if the Euclidean distance between predicted point and true target location, converted back to
  mm, is under a **20mm tolerance** — reusing `EVALUATION_STRATEGY.md`'s own settling-time
  tolerance radius as a ready-made, already-justified threshold rather than inventing a new one.
- **Aggregate:** hit-rate % per candidate per held-out session (§3), not pooled across sessions
  into one number — report the per-session distribution so a reader can see variance, not just
  a mean.

### 2.2 Metric B — Frozen-backbone near-future ball-position predictivity (linear probe)

The item's "predict the near-future from recent video" framing, made concrete and cheap. Does
**not** prompt the model in natural language for numeric coordinates (VLMs are known to be
unreliable at precise numeric regression via text generation) — instead probes the backbone's
own internal representation directly, which is both cheaper and a fairer test of what the
representation actually encodes.

- **Feature extraction (one forward pass per frame, no gradient updates to the backbone):** for
  each candidate, run its frozen vision encoder (or vision-language backbone up to a fixed
  pooling point — the same architectural attachment point item 3 designed for Qwen's action-query
  token, §3.1 of `MULTI_HEAD_ARCHITECTURE_SPEC.md`) over each sampled frame, extract one
  fixed-dimension pooled hidden-state vector per frame. Document the exact extraction point
  per candidate explicitly (it differs — Qwen2.5-VL's merger output vs. π0.5's PaliGemma
  embedding) rather than assuming a shared shape.
- **Two variants, kept separate (isolating the variable per skill §3):**
  - *Vision-only:* probe input = frame feature alone.
  - *Vision+state:* probe input = frame feature concatenated with the current
    `[touch_x_mm, touch_y_mm]` (2-dim, matching `convert_to_lerobot.py`'s real, already-built
    `observation.state` schema — not `RLControl.cpp`'s richer 9-dim contract, which the
    converter doesn't populate yet, per `MULTI_HEAD_ARCHITECTURE_SPEC.md` §2.1).
  Reporting both, separately, answers "does state help" without conflating it with "which
  backbone is better" — the same discipline this project's small-vs-medium vision comparison is
  still missing and deliberately hasn't resolved by assumption.
- **Target:** `[touch_x_mm, touch_y_mm]` at frame *t+N*. **N starting point: 10 frames at the
  session's native 30fps ≈ 333ms** — reusing `MULTI_HEAD_ARCHITECTURE_SPEC.md` §5.4's own
  chunk-size starting point (derived from the same 500ms settling-time window), not a fresh
  guess. Per that doc and skill §4, this is a starting point for experimentation, not a locked
  value — item 5 should sweep a small range (~5-15 frames) rather than commit to N=10 from one
  run.
- **Fit:** closed-form ridge regression (e.g. `sklearn.linear_model.Ridge`) from pooled feature
  → target, fit per candidate per variant on the fit-split frames (§3). This is deliberately
  **not** gradient-descent fine-tuning — a least-squares fit over pre-extracted frozen features
  takes seconds once features are extracted, which is what makes this a *bootstrap* comparison
  rather than a training run. The one-time cost is the forward-pass feature extraction per
  candidate (§4), not the probe fit itself.
- **For Jetson-PI/π0.5 specifically, add a third variant:** its own native flow-matching action
  expert's predicted next-state (from its already-trained, non-fine-tuned action head) scored
  the same way — giving a direct "purpose-built action head vs. linear probe on a plain VLM"
  comparison point that no other candidate here can offer.
- **Metric:** mean Euclidean error in mm on held-out sessions — same units/formula as
  `EVALUATION_STRATEGY.md`'s steady-state-error metric (directly comparable in spirit, though
  this is a representation-quality proxy, not a closed-loop control measurement, and should be
  reported and discussed as such, never conflated with a real steady-state-error number).

### 2.3 What these metrics deliberately do not claim

Neither metric predicts final closed-loop control quality — a backbone that grounds well and
predicts near-future ball position well from frozen features is a *necessary*, not *sufficient*,
condition for a good final policy (the actual action head, fine-tuning quality, and control-loop
integration all still matter and aren't tested here). Frame both results to whoever reads item
5's notebook output as "which backbone is worth investing further engineering time in," not
"which arm wins."

### 2.4 Explicitly descoped from this proxy

- Directional commands (`forward/left/right/backward/hold/stop`) — no single well-defined target
  point exists for them the way color commands have a marker location; scoring would need a
  qualitative direction-correctness judgment call this plan doesn't specify. A future extension,
  not built here.
- Any real robot/hardware-in-the-loop step. This proxy runs entirely offline against recorded
  Track 4 sessions.
- Fine-tuning of any kind (LoRA included) — that's item 3's design, still unimplemented, and
  explicitly item 5/6/later work, not this plan.

## 3. Data: which Track 4 sessions, and split concerns

Source: `host_software/ml_jetson_vla/data_processing/session_manifest.json` — 10
`session_jetson_track4_*` sessions, 40,713 frames total, zero corruption flags, zero null
ground-truth rows, `telemetry.csv` row count exactly equals video frame count in every session
(single-writer-thread design, confirmed by the manifest's own audit). This is the only regime in
the catalog with real recognized language labels *and* real RL-control-net-driven closed-loop
motion (`DATASET_CATALOG.md`'s own "strongest regime found" assessment) — the PID-regime
sessions are a different action distribution (PID vs. RL control net) and should not be mixed in,
matching that doc's explicit warning.

**Split: session-level, not frame-level — this is load-bearing, not a suggestion.**
`MULTI_HEAD_ARCHITECTURE_SPEC.md` §7 already flags why: adjacent frames within one continuous
`rgb_video.mp4` are highly correlated (same lighting, same approach trajectory), so a frame-level
split would put near-duplicate frames on both sides and silently leak. Both metrics above must
hold out whole sessions, never individual frames.

**Recommended procedure: leave-one-session-out across all 10 sessions (10 folds), not a single
fixed train/eval split.** Per `model-iteration-constraints` skill §4 (never rank from a small
seed count — this project has been burned twice by exactly this), a single held-out session is
one data point; report the metric's distribution across all 10 folds (mean + spread), not a
single number, before treating any candidate ranking as real. 10 folds comfortably clears the
skill's "≥5" floor.

**Per-command vocabulary coverage — a real, concrete constraint on Metric A's folds.** Not every
session contains all 4 color commands (from the manifest's own per-session notes):

| Session | Missing color command(s) |
|---|---|
| `..._160025` | no `go_black` |
| `..._161239` | no `go_yellow` |
| `..._163702` | no `go_green` |
| `..._164115` | no `go_green`, no `go_yellow` |
| all other 6 sessions (`151627`, `153053`, `153247`, `160509`, `161712`, `164743`) | full 4-color vocabulary present |

For Metric A specifically, when a held-out fold's session is missing a color, that fold simply
cannot evaluate hit-rate for that color — report per-color hit rate, not just an overall average,
so a missing color in one fold doesn't silently get treated as a 0% score for that color. Metric
B has no such constraint (near-future position prediction doesn't depend on which command is
active).

**Open limitation, flagged rather than fixed here:** all 10 sessions were collected on the same
day (2026-09-15) within roughly a 2-hour window (15:16-17:12, confirmed by
`DATASET_CATALOG.md`'s own file-timestamp check) — almost certainly the same physical rig
position, lighting, and marker layout throughout. Session-level leave-one-out controls for
temporal/trajectory autocorrelation *within* a session, but **cannot** control for a systematic
condition shared by all 10 sessions (e.g. today's specific lighting). Any comparison result from
this plan should be read as "bootstrap signal under today's single collection condition," not a
claim of robustness across lighting/rig setups — a genuine gap, not something session-level
splitting fixes.

**DVC note:** per this session's git history, Track 4 bronze sessions are a documented DVC
exception (tracked despite `01_bronze` normally staying outside DVC) — item 5 should `dvc pull`
these session directories rather than assume they're present as plain files in a fresh clone.

## 4. Resource / time cost estimate per candidate

| Candidate | Download size | Params | Load time | Memory footprint (inference, FP16/BF16) |
|---|---|---|---|---|
| Qwen2.5-VL-3B-Instruct | **0 (already on disk)** — 2 safetensors shards, confirmed 3.8GB + 3.3GB = **7.1GB total** on this machine | ~3.74B | **Unconfirmed — no recorded run exists.** Searched this repo for any prior `qwen_vl_smoke_test.py` output/log; none found. Do not treat any load-time number as measured until item 5 actually runs it and records the output; a rough community-benchmark placeholder for a ~7GB HF checkpoint is 30-90s on a modern datacenter GPU, longer on Jetson NVMe/eMMC — **explicitly a placeholder, not this project's own number** | ~7.5GB weights (matches on-disk bf16 size) + activation/KV-cache overhead on top; item 5's bootstrap run does **not** need INT4/Jetson deployment to execute this proxy — a single Colab GPU with ≥10-12GB VRAM (T4/A100-class) running FP16 is sufficient, since this is an offline proxy eval, not an on-Jetson deployment test |
| Jetson-PI / π0.5 (OpenPI) | **Not downloaded yet** — estimate only, by analogy to Qwen's per-param packing: ~6.5-7GB for the combined PaliGemma+action-expert checkpoint, **unconfirmed until actually pulled** | ~3.3B | Unconfirmed, no local checkpoint to test load time against yet | Estimate ~6-7GB weights in FP16, comparable to Qwen; this candidate is new integration work for item 5 (download + `openpi`/Jetson-PI's own loading code), not a zero-cost reuse the way Qwen is |
| NVIDIA Isaac GR00T N1 | **Unknown — gated.** Do not estimate download size before the §1 pre-check (variant size, license) resolves | Unconfirmed | N/A until gated | N/A until gated |
| SmolVLA (harness-validation only) | ~1-2GB (450M params) | 450M | Fast — minutes, not the tens-of-minutes class the 3-4B candidates likely need | Comfortably fits any Colab GPU tier; justification for including it at all: lets item 5 validate the probe-fitting/prompt-parsing harness code cheaply before spending expensive time running the same harness against Qwen2.5-VL-3B/π0.5. Not scored against them — a pipeline smoke test, not a comparison entrant |

**Overall scoping note for item 5:** Qwen2.5-VL-3B is the only candidate with zero new download
cost (already staged, already smoke-tested for basic `generate()` functionality). Jetson-PI/π0.5
is real new integration work (download + wiring its own loading/inference code into whatever
harness item 5 builds). GR00T N1 should not be budgeted for at all until its gate (§1) resolves.
If item 5 is time-constrained, Qwen2.5-VL-3B alone (both metrics, all 10 folds) is a complete,
self-contained bootstrap result on its own; π0.5 is the next priority once that's done, not a
parallel must-have from the start.

## 5. Open questions and unconfirmed assumptions (explicit, per this project's documentation convention)

1. **Qwen2.5-VL's grounded point/bbox output format against this exact local checkpoint and
   `transformers` version** — not verified this session. Item 5 must confirm the real output
   format (and write a real parser against it) before Metric A can run; the prompt text above is
   a starting point, not a tested one.
2. **Whether π0.5/Jetson-PI's architecture cleanly exposes a poolable frozen hidden state** for
   Metric B's vision-only/vision+state variants the way Qwen's merger output does — not checked.
   If it doesn't cleanly separate from the action-expert path, Metric B's probe variant may need
   a different extraction point for this candidate specifically.
3. **GR00T N1's smallest variant's real parameter count and license terms** — both unconfirmed,
   gating whether it enters the comparison at all (§1).
4. **The single-day/single-lighting-condition limitation on all 10 Track 4 sessions (§3)** — not
   fixable within this plan's scope; flagged so any comparison result is read with that caveat
   attached, not silently generalized beyond today's collection conditions.
5. **Whether the 20mm tolerance (Metric A) and ~333ms/N=10 window (Metric B) are the right
   proxy-specific values**, versus values re-derived specifically for a representation-quality
   proxy rather than borrowed from the closed-loop control metrics they're reused from — reusing
   `EVALUATION_STRATEGY.md`'s and `MULTI_HEAD_ARCHITECTURE_SPEC.md`'s existing, already-justified
   numbers was a deliberate choice to avoid inventing new unvalidated thresholds, but neither was
   originally derived for *this* purpose — worth revisiting if item 5's results look threshold-
   sensitive.
6. **Whether SmolVLA's inclusion as a harness-validation-only step is actually worth the extra
   scope**, versus item 5 simply testing its harness code directly against Qwen2.5-VL-3B first
   (which is already staged) and skipping SmolVLA entirely — a judgment call left to item 5,
   not resolved here.
