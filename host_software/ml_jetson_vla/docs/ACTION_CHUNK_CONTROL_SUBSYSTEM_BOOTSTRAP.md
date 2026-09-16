# Decoupled Fast-Control-Subsystem — Video-Bootstrapped Prototype (Track 4, item 7)

**Status: design + a real bootstrap run against all 10 Track 4 sessions, no hardware, no
firmware.** Code: `core/action_chunk_bootstrap.py`. Read this doc first; the code file's
own docstring restates the scope argument in short form for anyone who opens it without
this doc.

## 0. Is this actually distinct from Metric B (items 4/5)? Yes — resolved here

The kickoff prompt's own hypothesis was: *"Metric B is a scoring tool used to compare
candidate VLA backbones against each other — item 7 wants the actual decoupled
fast-control-subsystem design/prototype itself, using the same video-bootstrapped
validation shape, not another comparison metric."* Confirmed correct, with one
refinement worth stating precisely rather than just agreeing:

| | Metric B (`BOOTSTRAP_MODEL_COMPARISON_PLAN.md` §2.2) | This item (item 7) |
|---|---|---|
| **Question asked** | Whose frozen representation (Qwen2.5-VL-3B vs. π0.5) best predicts **one** future point? | Does the **chunk-and-execute-open-loop mechanism itself** hold up as the chunk horizon N grows toward the task's own "predict the next 1-2s" framing? |
| **Target shape** | Single point, `[touch_x, touch_y]` at `t+N` | A full sequence, `[touch_x, touch_y]` (or `[theta_a,b,c]`) at every step `t+1 .. t+N` |
| **Purpose** | Rank arm-2 **backbone candidates** against each other, before committing engineering time to one | Validate the **control-subsystem mechanism** (action chunking / open-loop execution) that any arm-2 candidate will need to use to hit real-time rates — independent of which backbone eventually wins Metric B |
| **What a bad result means** | "This backbone's representation is weak, don't invest further in it" | "Open-loop execution degrades too fast past chunk length N, tighten the replan rate or shrink N" |
| **Reused from** | Nothing upstream — this plan is the origin of the frozen-backbone-probe idea | `qwen_multihead_policy.py`'s already-designed Head B/C shapes (item 3), plus Metric B's own ridge-regression/LOSO methodology, extended from one point to a chunk |

Concretely: Metric B **cannot answer item 7's question at all**, even in principle — its
target is one point at one horizon, so it has no way to express "how does error grow
across a 30-frame open-loop chunk." A perfect Metric B score (backbone X nails the
single point at `t+10`) says nothing about whether that same backbone's chunk output at
step 25 of 30 has drifted into uselessness. These are complementary, not overlapping,
questions, and this item is not a re-run or rename of Metric B under a new banner.

**Net scope call: genuinely new work**, not a scoped-down validation of item 3/4 output —
though it deliberately *reuses* item 3's head shapes and Metric B's fitting methodology
rather than inventing either from scratch (see §2).

## 1. What "decoupled high-frequency control subsystem" means today (post-2026-09-15 redirect)

`LARGE_VLA_RESEARCH_SPIKE.md`'s 2026-09-15 revision is load-bearing here and easy to
misread if only the doc's *title* section ("dual-tier framing replaced") is skimmed: it
did **not** replace the fast tier with nothing — it replaced a **from-scratch custom
inner-tier model** with **action chunking inside a single existing action-chunking VLA**
(SmolVLA, primary candidate). The "decoupling" is not two models talking to each other;
it's one model that predicts N future actions per forward pass and lets the control loop
execute that chunk open-loop across N ticks, so tick rate is decoupled from inference
rate without a second network. This matches the user's explicit constraint quoted in the
kickoff prompt: *"we should research/use an action arm which already exists... do not
want to train/design anything from scratch... only fine tune it."* Item 7's prototype has
to honor that — it cannot design a new "fast tier" architecture, and it should validate
the **mechanism** (chunking), not invent an alternative to it.

## 2. Why not just run real SmolVLA/Qwen2.5-VL-3B for this bootstrap

Two independent reasons, both concrete, both already established elsewhere in this
track rather than asserted fresh here:

- **Resource.** `MULTI_HEAD_ARCHITECTURE_SPEC.md` §0/§1 already establishes this is a
  ~3.74B-param model, BF16-native, ~7.1GB on disk, with `deployment/`'s INT4 export path
  still "a scaffold — there is no model to point it at yet." `BOOTSTRAP_MODEL_COMPARISON_PLAN.md`
  §4 scopes even the *frozen-inference* feature-extraction pass (no training) at Colab
  A100-tier GPU (≥10-12GB VRAM). The task's own framing is explicit: "designed/prototyped
  at home... before any hardware test" — running the real backbone is exactly the kind of
  step that framing is asking to defer, not the step to build this prototype around.
- **It wouldn't isolate the right variable anyway.** Item 7's question is about the
  *chunking mechanism's* accuracy-vs-horizon behavior, not about which backbone's
  representation is best (that's Metric B's job, §0). Running the full Qwen backbone here
  would spend the expensive resource on the wrong axis — `model-iteration-constraints`
  skill §3 ("isolate the variable you're testing") argues directly against conflating the
  two the way a "just run Qwen for this too" shortcut would.

**What this prototype uses instead, honestly scoped as a surrogate, not a replacement:**
the already-vision-processed `touch_x_mm/touch_y_mm` (and `target_x/y`, `theta_a/b/c`)
signal already sitting in every Track 4 session's `telemetry.csv` — see `DATASET_CATALOG.md`'s
own "single-writer-thread, `rows_match_video: true`" finding, re-confirmed this session
(all 10 sessions, zero NaN ground-truth rows). A window of recent `touch_x/y` values *is*
"the last W frames of video, already processed by the real vision pipeline" — not a stand-in
for video, the vision pipeline's own already-extracted output. This is what makes the
whole exercise cheap enough to run on a laptop CPU in under two minutes (see §5) while
still being a real test against real recorded closed-loop motion, not synthetic data.

## 3. Two variants, reusing two different existing things, deliberately not conflated

Per `model-iteration-constraints` skill §3 (isolate the variable), these are kept as
separate code paths in `action_chunk_bootstrap.py`, not blended into one "average" score:

### 3.1 Ridge variant (primary) — reuses Metric B's own fitting methodology

Closed-form `sklearn.linear_model.Ridge`, multi-output (chunk flattened to `N*target_dim`),
fit per leave-one-session-out fold — literally Metric B's §2.2 recipe ("deliberately not
gradient-descent fine-tuning... a least-squares fit over pre-extracted features"), with
two changes: the target is a full future chunk instead of one point, and the "features"
are a window of `W` past state frames instead of one backbone's pooled hidden state.
Target: `[touch_x_mm, touch_y_mm]` per step — same units as Metric B and
`EVALUATION_STRATEGY.md`'s steady-state-error metric, directly comparable in spirit
(never conflated with a real closed-loop number, same caveat Metric B states).

Two feature-set variants (isolating "does knowing the upcoming target help," mirroring
Metric B's vision-only/vision+state split):
- `state_only`: `[touch_x, touch_y]` history only.
- `state_plus_target`: `[touch_x, touch_y, target_x, target_y]` history (default).

Window `W` swept over `{1, 5, 15, 30}` frames @30fps — `W=1` deliberately matches Head
B's real, already-designed single-frame state contract exactly (item 3 never assumed
multi-frame history), so this sweep also answers an open question item 3 didn't test:
does the task's own "given 10s of video" multi-frame framing actually help over a single
current-frame state, or is single-frame sufficient? Chunk `N` swept over
`{5, 10, 15, 30, 60}` frames — `5/10/15` covers item 3 §5.4's original ~333ms starting
range, `30/60` covers this task's own explicit "predict the next 1-2s" framing (a
materially longer horizon than item 3 ever scoped, worth stating as a genuine finding,
not glossed over).

### 3.2 Neural surrogate (secondary) — reuses item 3's Head B/C shapes verbatim

`ChunkSurrogatePolicy` in `action_chunk_bootstrap.py` imports and wires
`qwen_multihead_policy.py`'s `StateFusionHead` (Head B) and `ActionChunkHead` (Head C)
**unmodified** — same `Linear(state_dim,2048)+LayerNorm`, same
`Linear(4096,512)->ReLU->Linear(512,128)->ReLU->Linear(128,N*3)` with `tanh * 11.025°`
bounding. The only new component is `WindowEncoder`, a single `Linear` standing in for
Qwen's own pooled action-query hidden state — explicitly documented in-code as a cheap
stand-in, not a claim that a flattened few-scalar window carries Qwen's representational
power. Because `ActionChunkHead`'s bounding is degree-space (`tanh * ANGLE_LIMIT_DEG`,
designed for `RLControl.cpp`'s convention per spec doc §3.3), this variant always targets
`theta_a/b/c` — the same action convention item 3 committed to, not `touch_x/y` mm. This
directly answers the kickoff prompt's question of whether the prototype should
"reuse/validate [item 3's] specific design" — yes, and it does so literally (same
classes, same import), at one representative `(W=15, N=30)` setting rather than the full
sweep, to stay runnable on a CPU at home (a few dozen Adam steps per fold × 10 folds,
not a sweep × 10 folds).

## 4. Data and split discipline

Same 10 `session_jetson_track4_*` bronze sessions as items 4/5 (`session_manifest.json`),
read directly from `telemetry.csv` (no video decode — see §2). Re-verified this session:
all 10 have zero NaN rows across `touch_x/y/target_x/y/theta_a/b/c`, session lengths
2,294–5,193 frames (~76s–173s @30fps) — long enough for both the `W=30` (1s) history
window and the `N=60` (2s) chunk horizon to fit inside every session with room to spare.
`theta_a/b/c` measured range across all 10 sessions is exactly `[-11.025°, +11.025°]`,
confirming `qwen_multihead_policy.py`'s `DEFAULT_ANGLE_LIMIT_DEG` derivation against real
data rather than only against the firmware constant it was derived from.

**Split: session-level leave-one-session-out, all 10 folds, same discipline as Metric B**
(`BOOTSTRAP_MODEL_COMPARISON_PLAN.md` §3, `model-iteration-constraints` skill §4 — never
rank from a small seed count). Every reported number below is a mean ± spread across all
10 folds, never a single-session result. Same single-day/single-lighting-condition
caveat Metric B already flagged applies here identically — this is bootstrap signal under
2026-09-15's collection conditions, not a claim of robustness across rig/lighting setups.

## 5. Results — real run against all 10 sessions

Run command: `python core/action_chunk_bootstrap.py --run-neural` (defaults: `state_plus_target`
features, `touch_xy` ridge target, stride 2, ridge alpha 1.0, neural at `W=15 N=30`).
Full JSON output: `reports/action_chunk_bootstrap_20260915T221245Z.json` (per-fold curves,
not just the means summarized below). One real bug found and fixed during this run,
documented in the code's own docstring rather than only here: the first attempt at the
neural variant did full-batch gradient descent over ~36K rows through item 3's real
`Linear(4096,512)` fusion layer for 30 epochs × 10 folds — confirmed still running after
10 minutes / ~1.5 CPU-hours (`Get-CimInstance Win32_Process` against the real PID), an
estimated ~150 TFLOPs for one `(window, chunk)` setting. Fixed by mini-batching + a
4,000-row training subsample per fold (`train_cap`), bringing the same setting under two
minutes — see §5.2's honesty caveat on what that scoping costs the result.

### 5.1 Ridge variant — chunk-horizon error growth (mm, `touch_xy`, `state_plus_target`, 10-fold LOSO)

| W (frames) | N=5 | N=10 | N=15 | N=30 | N=60 |
|---|---|---|---|---|---|
| 1 (single-frame, matches Head B today) | step1 2.85 / mean 5.27 / stepN 7.25 | mean 7.26 / stepN 10.30 | mean 8.52 / stepN 11.33 | mean 9.72 / stepN 10.36 | mean 10.85 / stepN 12.82 |
| 5 | step1 2.87 / mean 5.38 / stepN 7.45 | mean 7.48 / stepN 10.67 | mean 8.72 / stepN 11.25 | mean 9.50 / stepN 9.99 | mean 10.65 / stepN 12.55 |
| 15 | step1 3.24 / mean 5.72 / stepN 7.55 | mean 7.37 / stepN 9.53 | mean 8.02 / stepN 9.02 | mean 8.33 / stepN 9.16 | mean 9.58 / stepN 11.58 |
| 30 (≈1s history) | step1 2.92 / mean 4.82 / stepN 6.27 | mean 5.95 / stepN 7.35 | mean 6.44 / stepN 7.39 | mean 7.08 / stepN 8.41 | mean 8.46 / stepN 10.96 |

(all values mm; `stepN` = error at the last frame of that chunk; full per-fold curves +
std in the JSON.) Four real findings, not glossed over:

1. **Step-1 error is essentially flat across everything tested (~2.85-3.27mm)** — chunk
   length and window length barely move the *immediate*-next-frame prediction, which makes
   sense (it's dominated by how noisy `touch_x/y` measurement itself is, not by the
   forecasting problem). Comfortably under `EVALUATION_STRATEGY.md`'s 20mm settling-radius
   tolerance, reused here only as a reference line (§6), not re-derived for this purpose.
2. **Longer history windows measurably help, and the effect grows with chunk horizon** —
   `W=30` beats `W=1` at every chunk length, and the gap widens as `N` grows (mean error at
   `N=60`: 8.46mm at `W=30` vs. 10.85mm at `W=1`, a ~22% reduction; at `N=5` the two are
   within noise, 4.82 vs. 5.27mm). This is a genuine finding item 3 never tested — Head B's
   real, already-designed contract is single-frame (`state_dim=2` at one timestep), and
   this result says that's leaving accuracy on the table specifically at the longer
   horizons this task's own "1-2s" framing cares about. Worth feeding back into whether a
   future Head B revision should carry a short window instead of one frame.
3. **`W=15`'s step-1/mean error is anomalously worse than both `W=5` and `W=30`** (3.24mm
   vs. 2.87mm and 2.92mm) — not a clean monotonic trend. Ridge `alpha` was held fixed at
   1.0 across the whole sweep, not tuned per window size; a wider input (more history
   frames feeding one under-regularized closed-form fit) plausibly overfits at some window
   sizes more than others. Flagged as unexplained, not swept further here — a real open
   item, not resolved by this bootstrap.
4. **Error is not always monotonic in `N`'s own `stepN` column** (e.g. `W=1`: `stepN` at
   `N=15` is 11.33mm but at `N=30` it's *lower*, 10.36mm) — each row of the table is a
   *separately fit* model for that chunk length, not one growing curve, so this compares
   different models' last-step error, not one trajectory's drift. A plausible physical
   explanation, not confirmed further here: the platform's own settling behavior
   (`EVALUATION_STRATEGY.md`'s ~500ms settling window) means a 1s-ahead prediction may
   often land after the ball has already re-settled near its target, which can be *easier*
   in absolute mm terms than a mid-transient 500ms-ahead prediction — worth checking
   against real per-episode settling data before treating this as a load-bearing result.

**Session-level outlier, load-bearing for how these means should be read**:
`session_jetson_track4_20260915_163702` is 3-4x worse than every other session at every
setting (e.g. at `W=1, N=10`: step1 8.51mm / stepN 32.31mm, vs. 1.9-2.5mm / 6.3-9.7mm for
the other 9 sessions — see the JSON's per-fold breakdown). Per
`model-iteration-constraints` skill §4, this is exactly why the design doc reports the
per-fold distribution rather than only a pooled mean: this one session's own dynamics
(more aggressive commands, faster ball motion, or something else not diagnosed here) are
pulling every `stepN`/mean figure above upward substantially; a reader should weight the
9-session-consistent numbers more than the outlier-inflated mean. Not investigated further
here (out of this task's scope) — flagged as a real, concrete follow-up.

### 5.2 Neural surrogate — item 3's real Head B/C shapes, theta-space, `W=15 N=30`

Result: **essentially flat across the whole 30-step chunk** — step1 = 3.72°, step30 =
3.76°, mean = 3.71°, std ≈1.1-1.2° throughout (full 30-point curve in the JSON). This
looks, at first read, like "the chunking mechanism has zero drift in theta-space" — but
that reading should be resisted: this variant was deliberately compute-scoped (§2, code
docstring) to a 4,000-row random subsample and 10 epochs of mini-batch training per fold,
specifically to stay CPU-runnable at home. A flat, ~3.7°-everywhere curve is at least as
consistent with **underfitting to something close to the training set's mean `theta`**
(which would trivially produce a constant-ish prediction and hence a flat error curve) as
with genuine no-drift chunking behavior. For scale: 3.7° is ~34% of
`DEFAULT_ANGLE_LIMIT_DEG` (11.025°, arm 1's own real operating envelope) — a real,
non-trivial error, not a "solved" result. **Net: this run validates that item 3's actual
Head B/C shapes wire up and train end-to-end against real data without shape errors or
NaNs (the thing this variant was actually scoped to check, per §3.2) — it does not, on
this compute budget, produce a trustworthy accuracy number for theta-space chunking.**
Getting a real answer needs either more training budget (more epochs/full data, i.e. real
GPU time) or a smaller/cheaper head shape purpose-built for this scale of surrogate —
neither done here, flagged as follow-up.

## 6. What this bootstrap does and does not license

**Does license:** a first read on how far into a chunk open-loop execution can go before
per-step error crosses a threshold worth caring about — §5.1's ridge numbers stay
comfortably under `EVALUATION_STRATEGY.md`'s 20mm settling-radius tolerance (reused as a
reference line, not re-derived) even out to `N=60` (~2s, mean 8.46mm at the best window
size), so the task's own "1-2s" framing is not obviously unrealistic for this platform's
dynamics, though error is clearly still climbing with horizon, not plateaued. It also
licenses one concrete, actionable finding item 3 didn't have: a multi-frame history window
measurably beats item 3's real single-frame Head B contract at longer horizons (§5.1
finding 2), worth carrying into any future Head B revision.

**Does not license:**
- Any claim about the real Qwen2.5-VL-3B/SmolVLA backbone's chunk accuracy — this ran a
  ridge/tiny-MLP surrogate on already-extracted state, not the real backbone (§2).
- The neural surrogate's flat ~3.7°-everywhere curve (§5.2) as evidence that chunked
  theta-space prediction has "solved" multi-step drift — it's at least as likely a symptom
  of this variant's deliberate compute-scoping (4,000-row subsample, 10 epochs)
  underfitting toward a near-constant prediction. Flagged explicitly, not quietly reported
  as a clean result.
- Any closed-loop control-quality claim — steady-state error, settling time, control
  effort, task success rate stay `evaluate_system_control.py`'s job, once a real policy +
  hardware run exists.
- Any claim this generalizes past 2026-09-15's single collection session (§4), or past the
  one outlier-flagged session's disproportionate pull on the pooled means (§5.1).

## 7. Open items, not resolved here

1. Whether the ridge/neural error growth found here (§5) should directly set the real
   fine-tuned SmolVLA/Qwen chunk size, or only bound it — the real backbone's own
   representation may extrapolate differently than this state-only surrogate; treat §5 as
   informative, not load-bearing, for that final choice.
2. A raw-pixel (rather than already-extracted `touch_x/y`) variant of this same harness,
   using a small frozen off-the-shelf CNN (e.g. the same frozen-ResNet18 pattern
   `RT1LiteVLA` already uses in `ml_multimodal/`) — would more literally match "given
   video" at the pixel level instead of the vision-pipeline's already-processed output.
   Not built here: `touch_x/y` already *is* that pipeline's real output for these
   sessions, and building a second, heavier feature path for the same validation question
   wasn't judged worth the added runtime for what it would add on top of §5's finding —
   flagged as a possible follow-up, not a gap in this task's own scope.
3. This harness's `WindowEncoder` (§3.2) is a placeholder for Qwen's real pooled hidden
   state; once a real fine-tuned checkpoint exists (items 4/5/6), the honest follow-up is
   swapping it for the real backbone output and re-running the exact same per-step-error
   methodology, not re-deriving a new one.
