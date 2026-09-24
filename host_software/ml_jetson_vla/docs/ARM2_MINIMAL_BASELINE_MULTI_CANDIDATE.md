# Arm 2 Minimal-Baseline Multi-Candidate Extension (2026-09-18)

> **ERRATUM, 2026-09-19 -- every accuracy number reported in this doc's original §7 (and the
> 2026-09-18 A/B result file `deployment/arm2_minimal_baseline_prompt_ab_scoring_20260918.json`) is
> INVALID as a measure of model accuracy.** The offline scorer had two bugs, found while building
> the Colab sweep (details and evidence in §8):
> 1. **Wrong ground-truth frame.** Predictions were converted to the manifest mm frame
>    (top-left origin) but compared directly against `telemetry.csv`'s `target_x`/`target_y`, which
>    are centre-origin with x mirrored (`target_x = W/2 - x_manifest`, `target_y = y_manifest - H/2`).
>    A perfect prediction would have scored 78-170 mm error; "0 hits" was guaranteed.
> 2. **Wrong pixel space for Qwen's answers.** Qwen2.5-VL answers in the coordinate space of the
>    image its vision tower actually sees (640x480 is downsized to 504x364 under this project's
>    `max_pixels`), not the raw frame's pixels; the scorer read them as raw pixels.
>
> Corrected, on the same 60 frames and the same model outputs (no model re-run): **baseline prompt
> 40/60 hits, 27.9 mm mean / 14.1 mm median error; oriented prompt 21/60 hits, 43.7 mm mean** -- not
> 0/60 and 184/170 mm. The earlier conclusion that the oriented prompt "helped" was an artifact; on
> corrected numbers it is worse than baseline on this sample. Corrected file:
> `deployment/arm2_minimal_baseline_prompt_ab_scoring_20260918_RESCORED_v2.json` (the original is
> kept untouched for provenance). Also corrected here: §2's InternVL row and §5's InternVL row, which
> claimed a native-`transformers` load path that does not exist for the 4B checkpoint.

**Status: real code + a real live verification run, not a design doc.** Extends
`ARM2_MINIMAL_BASELINE_SCOPE.md` (read that first -- this doc assumes its framing: Arm 2's
comparison axis is general-purpose/minimum-coding-effort deployment vs. Arm 1's
high-specialization purpose-built model, judged on the project's standard four metrics once a
real policy exists, per that doc's section 1). That doc scoped the minimal baseline around ONE
model, Qwen2.5-VL-3B. This doc extends the same "honest minimum" architecture (prompt in, parse
structured output out, no custom-trained heads, no fine-tuning) to support comparing **multiple**
general-purpose VLM candidates through shared infrastructure, and reports what was actually run
vs. only structurally checked.

**Everything in `ARM2_MINIMAL_BASELINE_SCOPE.md` still applies unchanged**: the firmware gap
(§3, still blocked, still not this doc's to resolve), the specialization-track parking (§5,
untouched), and the "no firmware changes, no fine-tuning" boundary. This doc does not relitigate
any of that.

## 1. Confirmed: which candidates even belong in "minimal baseline" scope

The minimal-baseline framing only makes sense for **general-purpose, promptable VLMs** -- models
trainable-free-at-deployment-time to answer an arbitrary structured-output question about an
arbitrary image, regardless of what robot (if any) they've seen. It does **not** make sense for
**embodiment-specific action models** -- this was stated as an assumption in this task's prompt,
and was researched for real (WebSearch, 2026-09-18) rather than accepted on say-so, per this
session's standing "verify, don't assume" practice.

**Candidates checked**: π0.5/Jetson-PI (`docs/LARGE_VLA_RESEARCH_SPIKE.md`'s own
already-researched candidate) and NVIDIA Isaac GR00T N1
(`docs/BOOTSTRAP_MODEL_COMPARISON_PLAN.md`'s conditional candidate) -- both already surveyed
elsewhere in this project for the *specialization* track, re-examined here specifically for
whether either has a legitimate zero-shot (no fine-tuning, no adaptation data) path to a genuinely
novel action space like this platform's `theta_a/b/c` step-space.

- **GR00T N1**: uses embodiment-specific encoders -- small MLP "projector" networks that map
  each robot's own state/action dimensionality into the model's shared embedding space. A
  review of the architecture found while researching this (Medium, "A review paper — GR00T N1")
  confirms this directly: cross-embodiment support is implemented via these per-embodiment
  projectors (also called identity embeddings/soft prompts in follow-up work), which must exist
  and be trained/fit for a given embodiment before that embodiment can be used at all. One
  survey source found during this research states plainly: **"Zero-shot GR00T N1.5 fails
  entirely on different robot embodiments."** There is no projector for a 3-DOF parallel
  ball-balancing platform's step-space anywhere in GR00T N1's released checkpoints -- running it
  against this platform zero-shot would not produce a slower or lower-quality answer, it would
  produce output with no defined mapping to this platform's actuators at all.
- **π0.5 / Jetson-PI**: `pi.website`'s own paper/blog describes its generalization as
  cross-*task*/cross-*environment* within a family of already-co-trained embodiments ("transfer
  physical behaviors from other robots... simpler robots that have one arm or no mobile base"),
  demonstrated on mobile manipulators cleaning kitchens/bedrooms in new homes -- new *scenes*,
  not a new *action space*. Nothing in the paper or `docs/LARGE_VLA_RESEARCH_SPIKE.md`'s own
  prior research claims π0.5 can be pointed at an arbitrary new actuator geometry
  (tilt-platform step-space) with no fine-tuning and produce meaningful output. Its own action
  tokenizer (FAST) and action expert were trained against specific robot action spaces (arm
  joint deltas, gripper state, base velocities) that share no structure with this platform's
  `[stepA, stepB, stepC]` contract.
- **No counterexample found.** The broader WebSearch for "zero-shot cross-embodiment transfer"
  surfaced several 2026 papers whose own framing confirms this remains an open, unsolved research
  problem rather than an available off-the-shelf capability: paper titles like "MOTIF: Learning
  Action Motifs for **Few-shot** Cross-Embodiment Transfer" and "LAP: ... Enables **Zero-shot**
  Cross-Embodiment Transfer" (a claimed exception, but describing a specific pre-training method
  applied *to* a new model, not a capability either GR00T N1 or π0.5 already ships) and "Cloak:
  Zero-Shot Cross-Embodiment Manipulation by Masking the End-Effector" (narrowly scoped to
  swapping end-effector *hardware* on an otherwise-similar arm, not a wholly different actuation
  geometry). One synthesis note found during research is directly on point: even papers claiming
  "true zero-shot transfer to unseen embodiments" in this space share a real limitation --
  e.g. one cited example "transfers across robot arms, but always with the same parallel-jaw
  gripper it was trained on." Nothing found supports zero-shot transfer to a categorically
  different action space (a 3-motor tilt-platform, not a gripper-equipped arm) the way this
  project's Arm 2 would need.

**SmolVLA** (`lerobot/smolvla_base`, the original Track 4 specialization-track candidate before
the project's 2026-09-18 redirect toward "large" general-purpose models) is excluded from this
list too, for **two independent reasons**, neither alone sufficient on its own to skip
documenting: (1) it is the same category of embodiment-specific action model as π0.5 -- a
flow-matching action expert trained on the LeRobot community dataset's own robots' action spaces,
with the identical zero-shot cross-embodiment problem argued above, not a general promptable VLM;
(2) at 450M params it does not represent the "large model" class this track's comparison axis is
about (the user's explicit redirect earlier in this session). Either reason alone would exclude
it from this candidate list; both apply together.

**Conclusion, confirmed not just assumed**: π0.5/Jetson-PI, GR00T N1, and SmolVLA stay excluded from this
minimal-baseline candidate list, for the reason this task's prompt already anticipated -- running
either zero-fine-tune against this platform would produce output in a foreign, untrained action
space with no defined mapping to `theta_a/b/c` or step targets, not a fair "minimum effort"
comparison entrant. They remain exactly where `docs/LARGE_VLA_RESEARCH_SPIKE.md` and
`docs/BOOTSTRAP_MODEL_COMPARISON_PLAN.md` already put them: real candidates for the separate,
parked *specialization* track (fine-tuned action-chunking VLA), untouched by this doc.

Sources (WebSearch, 2026-09-18): [GR00T N1 review — Medium](https://medium.com/correll-lab/a-review-paper-gr00t-n1-an-open-foundation-model-for-generalist-humanoid-robots-nvidia-march-734c82c38e70), [NVIDIA/Isaac-GR00T](https://github.com/Nvidia/Isaac-GR00T), [pi.website — A VLA with Open-World Generalization](https://www.pi.website/blog/pi05), [π0.5 paper](https://www.pi.website/download/pi05.pdf), [MOTIF: Learning Action Motifs for Few-shot Cross-Embodiment Transfer](https://arxiv.org/pdf/2602.13764), [LAP: Language-Action Pre-Training Enables Zero-shot Cross-Embodiment Transfer](https://arxiv.org/pdf/2602.10556), [Cloak: Zero-Shot Cross-Embodiment Manipulation by Masking the End-Effector](https://arxiv.org/html/2606.22836v1).

## 2. Candidate shortlist -- general-purpose promptable VLMs only

Researched for real (WebSearch, 2026-09-18) rather than recalled from memory, per this
project's precedent (the Jetson-PI latency correction earlier this session). All four fit the
~4B realistic ceiling (`ARCHITECTURE.md`'s "Realistic model ceiling" row).

| Candidate | Params | License | Native grounding output format (confirmed real) | Status this session |
|---|---|---|---|---|
| **Qwen2.5-VL-3B-Instruct** | ~3.74B | **Qwen RESEARCH LICENSE** (not Apache -- confirmed via the checkpoint's own `LICENSE` file / HF discussion thread; flagged since this differs from the other three's permissive licenses and would need checking before any non-research use) | JSON list: `{"point_2d":[x,y],"label":...}` or `{"bbox_2d":[x1,y1,x2,y2],"label":...}`, absolute pixel coords | **Already downloaded, already live-verified this session** (§4) -- the confirmed baseline. |
| **InternVL2.5-4B** (`OpenGVLab/InternVL2_5-4B`) | ~4B (achieves ~90% of larger-variant performance at 5% of the size, per its own model card) | MIT (project license) + Apache 2.0 (its Qwen2.5-3B-Instruct LLM component) -- both permissive | `<ref>`/`<box>` referring-expression tags. **CORRECTED 2026-09-19:** this checkpoint is a `trust_remote_code` repo with its own `model.chat()` API; there is NO official HF-native (`-hf`/`-HF`) conversion of the 4B (official ones: 2B-MPO-hf, 8B-MPO-hf, InternVL3-*-hf, InternVL3_5-*-HF), so the original "native `InternVLForConditionalGeneration`" claim and load call were wrong | Never executed -- see §5, §8. |
| **PaliGemma2-3B-mix-448** (`google/paligemma2-3b-mix-448`) | 3B | Gemma license -- permits redistribution, commercial use, fine-tuning | `<locY1><locX1><locY2><locX2>` special tokens, normalized to a 1024-cell grid -- **not JSON**, and NOT a free-form-chat model (task-prefix prompts like `"detect {object}"`, confirmed via WebSearch) | Structural only -- see §5. |
| **Moondream2** (`vikhyatk/moondream2`) | 1.9B -- smallest by a wide margin | Permissive (free for personal/research/most commercial use per its own model page) | **Native structured API, no text parsing needed**: `model.point(image, label)["points"]` -> `[{"x":0-1 float,"y":0-1 float}, ...]` | API surface verified via real `config.json`/`README.md` fetch (small, not the full weights) -- see §5. |

**Considered and deprioritized**: **Phi-3.5-vision-instruct** (4.2B, MIT license, Microsoft) --
real, permissively-licensed, right size class. Deprioritized because WebSearch found no
confirmed native grounding/bounding-box output convention the way the other three have; its
documented strength is working *alongside* a separate object detector (reading labels a
detector already found) rather than emitting coordinates itself. Since this task's whole
prompt/parse contract (§3) depends on eliciting a point/box from the model directly, Phi-3.5-vision
would need a materially different (and unverified) prompting strategy to even attempt scoring --
recorded here as a real, deliberately-filtered-out candidate, not silently omitted.

**Net: 3 general-VLM alternatives to Qwen2.5-VL-3B** (InternVL2.5-4B, PaliGemma2-3B-mix,
Moondream2), within this task's requested 2-4 range, chosen to cover real architectural/API
diversity (native JSON grounding vs. native `<loc>`-token grounding vs. native structured
pointing API) rather than three near-identical Qwen-family variants.

## 3. Prompt / parse contract

Full design lives in `core/minimal_vlm_policy.py`'s module docstring (read there for the
complete reasoning) -- summarized here:

- **Ask**: point to the pixel location of the instruction's target (a color marker) in the
  image, as `{"target_point_xy": [x, y]}`. Chosen over asking for `theta_a/b/c` motor angles
  directly (which `ARM2_MINIMAL_BASELINE_SCOPE.md` S3 also allows) because a general VLM has
  zero training signal for this platform's motor geometry -- pointing to something visible is
  the actual capability general VLMs are trained for, and it is what makes offline scoring
  against real logged data (§6) possible without inventing a new ground-truth signal.
- **Parse**: `parse_minimal_baseline_output()` tries, in order: (1) a backend's native
  structured output if it has one (Moondream2's `.point()` result, no text parsing needed at
  all); (2) the JSON contract asked for above; (3) Qwen's native `point_2d`/`bbox_2d` JSON list
  convention, in case the model answers in its own native format instead of the asked-for one;
  (4) PaliGemma2's native `<loc>`-token convention. Anything else: `ok=False`, the raw text is
  preserved, and the caller gets `float("nan")` targets rather than a fabricated point --
  never silently guessed. This is a deliberate, documented choice (see the module docstring's
  "on parse failure" note), not an oversight.
- **Directional/hold/stop commands**: no well-defined target point exists for a general VLM
  with no spatial-direction convention for this platform -- the exact same reasoning
  `docs/BOOTSTRAP_MODEL_COMPARISON_PLAN.md` §2.4 used to descope these from its own Metric A.
  This wrapper mirrors that precedent: the prompt falls back to "point to the ball's current
  position" (a well-posed grounding ask) and every such frame is tagged
  `debug["directional_fallback"] = True` so it's never mistaken for a working
  directional-understanding result.

### 3.1 Open gap (2026-09-23): joint ball + marker prediction not in this contract

The user's actual stated goal is for the vision output/understanding framework to predict
**both** the ball's location **and** the colored markers' locations. That's already true and
locked for Arm 1 -- CLAUDE.md's Architecture Decisions table specifies the Shared Backbone CNN
as "one CNN, two heads (Ball + Markers)," built and working per the Current State table.

It is **not** true of this minimal-baseline contract. §3 above asks each candidate for exactly
**one** point per call: `target_label` resolves to either `"ball"` (directional/hold/stop
fallback) or a single named color marker (`"red marker"` etc., for `go_*` commands) -- never
both, and never all four markers at once (`core/minimal_vlm_policy.py:314-347`). A `go_red` frame
never asks where the ball currently is; a `hold`/`stop` frame never asks where the markers are.

This is a real, undecided scope gap, not a design that was considered and rejected. Closing it
would mean either (a) a second prompt/call per frame for the complementary target, doubling
latency-per-frame for every candidate, or (b) a redesigned single-call prompt/parse contract
asking for all five points (ball + 4 markers) at once, which every candidate would need
re-running against for a fair comparison (including Qwen2.5-VL-3B's already-complete 60-frame
sweep). Neither has been chosen. Flagged here per explicit user instruction (2026-09-23) to
document the gap without disrupting the PaliGemma2 sweep in progress on the Jetson at the time
this note was added.

## 4. Code infrastructure -- file paths

| File | What it is |
|---|---|
| `core/vlm_backends.py` | `VLMBackend` protocol + one class per candidate (`QwenVLBackend`, `InternVLBackend`, `PaliGemma2Backend`, `Moondream2Backend`) + `BACKEND_REGISTRY` (name -> class, for config/CLI selection). Each backend's docstring states plainly what was/wasn't verified against a real installed API this session (see §5). |
| `core/minimal_vlm_policy.py` | `MinimalVLMPolicy(Policy)` -- the model-agnostic wrapper. `act(image, instruction, state) -> PolicyCommand`, same `Policy` protocol shape `core/policy_interface.py` already defines and `JetsonExpertPolicy` already implements. Swapping candidates is `MinimalVLMPolicy(backend=<any backend>)`, a constructor argument. Also holds `PROMPT_TEMPLATE`, `build_prompt()`, `parse_minimal_baseline_output()`. |
| `deployment/qwen_vl_smoke_test.py` | **Modified, not replaced** -- `load_qwen_vl_model()`/`generate_qwen_vl()` factored out of `main()` into importable functions (`QwenVLBackend` imports them) so the already-working loading path is reused, not duplicated. CLI behavior unchanged. **Also fixes a real, previously-flagged-but-unfixed bug**: `docs/MULTI_HEAD_ARCHITECTURE_SPEC.md` §6 already found that this script hardcoded `torch_dtype=torch.float16` while the checkpoint's own `config.json` declares `bfloat16` -- a genuine bf16/fp16 mismatch risk that doc explicitly left unfixed ("not fixed here, this doc's scope is the design"). Fixed here: `load_qwen_vl_model()` now loads in `bfloat16` (matching the checkpoint's native dtype) on both CUDA and CPU. |
| `deployment/internvl25_smoke_test.py`, `deployment/paligemma2_smoke_test.py`, `deployment/moondream2_smoke_test.py` | Per-candidate smoke tests, same `--image`/`--camera_id`/`--prompt`/`--model_dir` CLI shape as `qwen_vl_smoke_test.py`. Each documents exactly what was verified about that candidate's real API this session vs. what a future run on a machine with the checkpoint would confirm live. |
| `deployment/score_minimal_baseline_offline.py` | Offline parse-rate/localization-accuracy scorer against real logged Track 4 sessions (§6). |

## 5. What was actually verified per candidate -- real API vs. executed

Per this task's explicit "be honest about what you could vs. couldn't actually execute"
instruction:

| Candidate | API verified how | Live weights executed this session? |
|---|---|---|
| Qwen2.5-VL-3B-Instruct | Already-working `qwen_vl_smoke_test.py` code, now also exercised through the new wrapper | **Yes** -- see §7, real output shown. |
| InternVL2.5-4B | ~~`hasattr(transformers, "InternVLForConditionalGeneration")` confirmed `True`~~ -- that check only proved the CLASS exists in the library, not that this checkpoint loads through it; **it does not** (see §2 correction). Rewritten 2026-09-19 against the model card's own `AutoModel(trust_remote_code)` + `model.chat()` API | No. Never executed as of 2026-09-19 either; first execution is the Colab sweep (§8). |
| PaliGemma2-3B-mix | `hasattr(transformers, "PaliGemmaForConditionalGeneration")` confirmed `True`, same machine/version; native `<loc>`-token output convention confirmed via WebSearch, not the installed API directly (no local checkpoint to introspect) | No -- same reasoning as InternVL2.5-4B. |
| Moondream2 | `config.json` (`auto_map`/`architectures`) and `README.md` (`.point()`/`.detect()`/`.query()` real call shapes, including the `.point()` normalized-[0,1] coordinate convention, confirmed via a follow-up WebSearch since the README snippet fetched didn't state it) fetched directly from the real `vikhyatk/moondream2` HF repo -- genuinely real, not guessed, but a config/readme fetch, not a weights download | No -- full ~4GB weights not downloaded/run this session. |

## 6. Run/test plan -- what's achievable now vs. blocked

- **Achievable now (done for Qwen, scaffolded for the others)**: a deployment-feasibility smoke
  test -- load + prompt + parse, real latency/memory numbers. See §7 for Qwen's real numbers;
  `deployment/internvl25_smoke_test.py`/`paligemma2_smoke_test.py`/`moondream2_smoke_test.py`
  are ready to run the same way on a machine with each checkpoint (or internet access to pull
  it), not yet run themselves.
- **Achievable now, offline**: `deployment/score_minimal_baseline_offline.py` scores the
  minimal-baseline prompt/parse path (not a separate probe metric -- see that script's module
  docstring for how this differs from `BOOTSTRAP_MODEL_COMPARISON_PLAN.md`'s Metric A) against
  real logged Track 4 `telemetry.csv`/`rgb_video.mp4` data: parse success rate, and localization
  error in mm (predicted pixel point -> mm via a real per-frame ArUco homography, compared
  against the session's own logged `target_x`/`target_y`). A real, small run of this is reported
  in §7.
- **Blocked on firmware**: real closed-loop motor execution and the project's standard four
  metrics (steady-state error, settling time, control effort, task success rate). Stated plainly
  as blocked, per `ARM2_MINIMAL_BASELINE_SCOPE.md` §3 -- not attempted, not worked around.

## 7. Verification results (real, 2026-09-18)

**Environment note, stated up front**: this session's sandbox has **no CUDA** (`torch.cuda.is_available()
== False`) -- unlike the target Jetson AGX Orin hardware `qwen_vl_smoke_test.py` was originally
confirmed against. All latency numbers below are **CPU-only, on a 12-core dev machine**, not
representative of real Jetson performance -- they confirm the wrapper/prompt/parser work
end-to-end against the real model, not that this is what Jetson deployment will feel like.
Loading succeeded in bf16 on CPU (~10-13s); a full `generate()` call is the real bottleneck.

### 7.1 Wrapper smoke run (real, through `MinimalVLMPolicy`/`QwenVLBackend`)

A single real frame from `session_jetson_track4_20260915_151627` (frame 60), prompted with
this module's `PROMPT_TEMPLATE` asking for the green marker's location: model responded
`{"target_point_xy": [293, 104]}` -- clean, directly-parseable JSON, matched by the
`asked_json_contract` branch of the parser on the first try (no fallback convention needed).
Load: 12.3s. Generate (32 tokens): 113.2s.

### 7.2 Offline scoring run (real, `deployment/score_minimal_baseline_offline.py`) -- **INVALID accuracy numbers, see the erratum at the top and §8**

*The parse-rate results and latencies below stand; every `hit_rate`/`mean_error_mm`/`error` figure
in this subsection was computed against the wrong ground-truth frame and (for Qwen) the wrong pixel
space, and must not be cited.* Kept as originally written for the record:

Command actually run:

```
python deployment/score_minimal_baseline_offline.py \
  --sessions session_jetson_track4_20260915_151627 session_jetson_track4_20260915_153053 \
  --max-frames-per-session 2 --max-new-tokens 24
```

4 frames total (2 sessions x 2 frames each), real color-command frames (`go_green`,
`go_yellow`, `go_green`, `go_red`), real per-frame ArUco homography, real logged
`target_x`/`target_y` ground truth from each session's `telemetry.csv`. Full per-frame JSON
kept alongside this doc's session artifacts.

| Session | frames | parse_success_rate | hit_rate (20mm) | mean_error_mm | mean generate latency (s) |
|---|---|---|---|---|---|
| `...151627` | 2 | **1.0** (2/2) | 0.0 (0/2) | 186.0 | 144.5 |
| `...153053` | 2 | **1.0** (2/2) | 0.0 (0/2) | 190.6 | 122.6 |

**Real per-frame example** (frame 1, `session_jetson_track4_20260915_151627`, instruction
`go_green`): raw output `{"target_point_xy": [297, 118]}`, parsed cleanly on the first
convention tried, converted via that frame's real ArUco homography to `(141.3, 127.9)` mm --
vs. the session's real logged target `(0.42, 29.61)` mm. Error: 171.8mm, well outside the
20mm tolerance.

**Honest reading of this result, not oversold**: parsing worked perfectly (4/4, 100%) --
the prompt/parser infrastructure itself is solid. Zero-shot localization accuracy was poor at
this n=4 sample (0/4 within 20mm; mean error ~180-190mm, roughly the same order of magnitude
as the platform's own physical half-width). This is a real, reportable data point in exactly
the direction `ARM2_MINIMAL_BASELINE_SCOPE.md` §4 already told this task to expect and report
plainly rather than engineer around ("if the minimal baseline['s]... is slow and jittery
compared to Arm 1, that gap IS the comparison's answer, not a defect in the baseline") --
the same logic extends to accuracy here. **Per the `model-iteration-constraints` skill's own
reminder this script prints at the end of every run: this is a 4-frame smoke score, not a
statistically powered accuracy measurement.** It does not establish Qwen2.5-VL-3B's true
zero-shot marker-localization accuracy on this platform (that needs the full leave-one-
session-out protocol `docs/BOOTSTRAP_MODEL_COMPARISON_PLAN.md` §3 already specifies, at a
frame count this session's CPU-only sandbox could not realistically run -- ~144s/frame at
24 tokens means even one full 10-fold pass would take hours on this hardware, let alone a
real Jetson-latency-representative run). It DOES establish, for real: (a) the wrapper/parser
pipeline works end to end against the real model with zero fabrication, (b) latency on CPU is
consistent with `ARM2_MINIMAL_BASELINE_SCOPE.md` §4's own prediction ("multi-second range,"
here multi-*minute*, CPU being slower than the Jetson's Tensor Cores this was originally
scoped against), and (c) there is a real, non-hypothetical accuracy gap worth tracking once a
larger run is affordable -- not asserted as proven from 4 frames.

### 7.3 Other candidates

InternVL2.5-4B, PaliGemma2-3B-mix, Moondream2: not executed this session (§5). Their smoke
test scripts (§4) are ready to run on a machine with GPU access and/or the checkpoints
downloaded; nothing about their code paths was structurally validated by an actual run the way
Qwen's was.

## 8. 2026-09-19: scoring corrections, `oriented_aruco`, and the Colab sweep sandbox

Follow-up to §7's "local CPU is too slow (~95-200s per call)": the sweep moves to Colab GPU
(`deployment/arm2_colab_sweep.ipynb`, engine in `deployment/colab_sweep.py`), 4 candidates x 3 prompt
variants. Building it surfaced the scorer bugs in the erratum at the top of this doc.

### 8.1 What was wrong, with the evidence

**(a) Ground-truth frame.** `telemetry.csv` `target_x/target_y` are centre-origin with x mirrored;
the scorer compared homography mm (manifest frame, top-left origin) against them directly.
Evidence: (i) `touch_x`/`touch_y` in the same CSV reach -88.35 on a 187.5 mm platform (a top-left
origin cannot be negative), and the logged targets sit at ~(+/-30, 0), (0, +/-30) -- a plus pattern
around the centre; (ii) `auto_label_shared_vision.py` already documents `ball_x_mm = W/2 - touch_x`,
`ball_y_mm = H/2 + touch_y` (2026-08-12 point-reflection fix); (iii) empirically, HSV-detected marker
centroids from 25 frames across all 10 sessions, projected through the per-frame ArUco homography,
match `(W/2 - x, y - H/2)` to a median of ~1 mm (0.3-7 mm), versus 78-170 mm under the old
comparison. (The earlier "cross-check" in `minimal_vlm_policy.py` that concluded top-left was wrong:
`(0.42, 29.61)` is 29.6 mm from the centre, not "near an edge".) Fix:
`score_minimal_baseline_offline.touch_frame_to_manifest_mm()`.

**(b) Qwen's coordinate space.** Qwen2.5-VL answers in the resized image the vision tower sees:
qwen_vl_utils resizes 640x480 -> 644x476, then the processor (this backend's `max_pixels=256*28*28`)
-> **504x364** (measured from the real processor's `image_grid_thw`, not assumed). Reading those
answers as raw pixels scaled every prediction ~27% / 32% too large in x / y. Evidence: no answer in
120 samples exceeded 504 in x or 364 in y (max 429 / 242); rescoring the same outputs in that space
moves the baseline from 1/60 to 40/60 hits. Fix: each backend now declares `coord_space`; Qwen's
`BackendOutput` reports `model_input_hw` measured per call; `to_raw_px()` rescales.

### 8.2 Corrected local result (Qwen2.5-VL-3B, CPU bf16, same 60 frames, same model outputs)

| | old (2026-09-18) | GT-frame fix only | both fixes (**valid**) |
|---|---|---|---|
| baseline: hits @ 20 mm / mean / median | 0/60 / 183.8 / - | 1/60 / 66.0 / 57.9 | **40/60 / 27.9 / 14.1** |
| oriented: hits @ 20 mm / mean / median | 0/60 / 170.3 / - | 1/60 / 61.9 / 58.1 | **21/60 / 43.7 / 41.1** |

Baseline beats oriented on mean error in **10/10 sessions** (the earlier "oriented helped in 9/10"
was the artifact). Reference yardsticks on the same frames: constant platform-centre guess **25.2
mm mean, 9/60 hits**; uniform-random point on the platform **~67 mm mean, ~5% hits**. So the
baseline prompt localizes markers clearly better than chance, but its *mean* error is not better
than simply guessing the centre (targets sit ~30 mm from centre) -- its median (14 mm) and hit rate
are. 6 of the 60 sampled frames are "transition" frames whose logged target has not yet settled to
the commanded marker (the setpoint ramps over a few frames, e.g. go_red target_x 0.0 -> 31.2);
reported as a secondary "settled-only" view (baseline, 54 frames: 74% hits, 23.6 mm); sampling is
unchanged. n = 60 (6 per session); per-session baseline mean error ranges 10.6-53.5 mm. Not a
powered ranking.

### 8.3 `oriented_aruco` prompt (third variant)

`core/minimal_vlm_policy.py` `PROMPT_VARIANTS["oriented_aruco"]` = the oriented prompt with its
"ignore the ArUco markers" sentence replaced by "never answer with one of them" plus a paragraph
generated at import from `hardware/platform_templates/ground_truth_manifest.json` (6 markers, IDs,
centres, 22.5 mm size, 187.5 x 142.0 mm platform, origin/axes convention). Prompt text only -- the
model still receives the raw frame; output contract and parser unchanged; `baseline`/`oriented`
untouched. Two things it contains that are **not** in the manifest, and one that is absent:
- **Camera orientation.** In every Track 4 frame the camera image is the platform rotated 180 degrees
  from the manifest frame (manifest ID 0, "top-left-corner", appears at the image's *bottom-right*;
  verified by ArUco detections in all 60 sampled frames, ~11-15 px scatter). Giving the model
  manifest-frame roles without saying so would point it at the wrong corners, so the text states the
  rotation and where each ID appears. This is rig geometry, not per-frame information, but it is a
  judgement call worth a second look.
- The manifest's `features` list is **empty** -- it holds no coloured-marker positions. They are
  deliberately not in the prompt either (that would hand over the answer).

Possible follow-up (not built): a millimetre-output variant asking the model for platform
coordinates instead of pixels.

### 8.4 Static findings on the never-executed candidates (fixed in `core/vlm_backends.py`)

- **InternVL2.5-4B**: remote-code repo with `model.chat()`; the 2026-09-18 native-class load would
  have failed. Rewritten to the model card's API and dynamic-tiling preprocessing (`max_tiles=6`).
  `InternVLNativeBackend` (`OpenGVLab/InternVL3_5-4B-HF`, a *different generation*) is kept as an
  optional fallback only. Its remote code targets old transformers; smoke mode will show whether it
  runs.
- **Moondream2**: remote code at the README-recommended `revision="2025-06-21"` (pinned as commit
  `9a7d4024...`) is reported broken on transformers>=5 (`all_tied_weights_keys`), so the notebook
  installs `transformers<5`. `.point()` -> `{"points":[{"x","y"}]}` normalized to [0,1] is confirmed
  in the model's own source at that revision. The README load call passes no dtype, so the default is
  fp32.
- **PaliGemma2 (gated)**: pixel values now cast to model dtype; `<loc>` tokens no longer depend on
  `skip_special_tokens`; the parser accepts several `<loc>` boxes (first used). Auth via Colab secret
  `HF_TOKEN`; skipped gracefully if not authorized.
- **Prompt-variant meaningfulness.** PaliGemma2 `detect <label>` and Moondream2 `.point()` ignore the
  prompt text, so a 3-variant comparison on them would be three identical runs. The sweep therefore
  runs each in two modes: `:prompt` / `:query` (prompt text reaches the model; the A/B/C comparison)
  and `:detect` / `:point` (native API, run once).

### 8.5 What was verified locally vs. what only Colab can verify

**Verified (real, run on this machine)**: corrected scoring reproduces the historical 183.8 / 170.3
mm from the raw outputs and gives 27.9 / 43.7 under the fixes; `deployment/test_colab_sweep_mock.py`
(31 checks) drives the real decode + homography + prompts + parser + scoring + checkpoint through mock
backends emitting all four native formats: oracle markers score ~3 mm through the corrected path
(~160 mm under the old comparison), Qwen-style resized-space answers are handled, interrupt/resume
loses nothing and skips finished candidates without loading them, load/call failures and gating are
isolated, and the validation gate both fails and passes correctly. The notebook's 16 code cells
execute top to bottom in mock mode from the extracted bundle zip alone (imports asserted to come from
the bundle), and a second pass resumes with nothing recomputed; `nbformat.validate` + `py_compile` on
every cell and the concatenation.

**Not verified (needs Colab/GPU)**: that InternVL2.5, PaliGemma2 or Moondream2 load or produce
parseable output at all; their coordinate conventions; whether Qwen on Colab (transformers<5, GPU,
fp16/bf16) reproduces the local outputs (the notebook's validation gate tests exactly this); GPU
memory/latency; the Moondream/InternVL remote code on the installed transformers; the HF gating/auth
flow; Drive I/O.

## 9. 2026-09-22: running the same sweep directly on the Jetson (no Colab)

The user is at the device, so the sweep can run on the actual deployment hardware (the Orin's Ampere GPU,
not Colab's T4/A100) against the 10 `session_jetson_track4_*` sessions already on disk.
`deployment/run_arm2_sweep_jetson.py` is a plain-script stage sequencer over the **unchanged**
`colab_sweep.py` engine (same 4 candidates x 3 prompt variants, same 6 frames/session, same corrected v2
scorer, same fault isolation, same validation gate, same per-call atomic checkpointing -- to local disk).
Full design, the environment-isolation decision (two environments: the existing Qwen one left alone, one
clone with `transformers>=4.56,<5` for the other three) and the exact ordered commands are in
`JETSON_ARM2_SWEEP_LOCAL.md`. Dry-run-verified with mock backends only
(`deployment/test_run_arm2_sweep_jetson_mock.py`); no real model has run through it yet. The Colab
notebook remains a working alternative and was not modified.

## 10. 2026-09-23: real PaliGemma2-3b-mix-448 results on the Jetson (`jetson_run1`, results2)

Full sweep completed after fixing the gated-access/corrupted-cache issue (§9's environment). Real
numbers, `arm2_jetson_sweep_results2/aggregate_table_jetson_run1.csv`, same 60 frames/scorer as Qwen's
§8.2 result:

| variant | parse_rate | hit@20mm | mean_err_mm | median_err_mm |
|---|---|---|---|---|
| `:prompt` baseline | 0/60 | -- | -- | -- |
| `:prompt` oriented | 0/60 | -- | -- | -- |
| `:prompt` oriented_aruco | 0/60 | -- | -- | -- |
| `:detect` (native `detect {label}`, prompt ignored) | 9/60 (15%) | 2/9 (22%) | 113.5 | 122.3 |
| *[ref] constant platform-centre guess* | *60/60* | *9/60 (15%)* | *25.2* | *29.6* |

**`:prompt` mode: total failure, not a parsing bug.** Confirmed live during the run (raw outputs
included `'unanswerable'` and degenerate token loops like `'{ x: 0, 0, 0, 1, 1, 1'`) — the
prompt/image are reaching the model correctly (ruling out a wiring issue), but this checkpoint is
task-prefix-trained (`"detect <thing>"`, `"answer en <question>"`), not instruction-following, so it
cannot produce the asked-for free-form JSON at all. 0/60 across all three variants is the expected
result of asking a non-chat model to follow a chat-style instruction, not a bug to fix.

**`:detect` mode (the fair, native-API comparison): also genuinely poor**, and not just because of
the low parse rate. The 9 frames it *did* parse average 113.5mm error / 22% hit rate — **worse than
the reference constant-centre guess** (25.2mm / 15% hit, and that reference "hits" purely by the
platform's targets clustering near its own centre, with zero vision). This is a real negative result
for this checkpoint on this platform (small top-down 640x480/448x448-resized frame, small colored
circular markers), not an infrastructure problem — same category of finding as Qwen's real §8.2
numbers, just the opposite direction.

Not yet investigated: whether `detect {target_label}`'s exact phrasing (e.g. `"detect red marker"`
vs. a more canonical single-noun form) affects the 15% parse rate, or whether the 448x448 resize is
disproportionately hurting localization of markers that are already small in the raw 640x480 frame.
Flagged, not pursued — three more candidates (InternVL2.5-4B, Moondream2 `:query`/`:point`) are still
pending and take priority per the existing run plan.

### 10.1 2026-09-23: per-frame follow-up — `:detect`'s 51 unparsed frames are a distinct failure mode, not the same small-object gap as the 9 that parsed

Read the real per-frame scorer JSON (not just the aggregate table) for both results directories:

- **All 51/51 unparsed `:detect` frames have a literally EMPTY `raw_text`** — not garbled or
  off-convention text, nothing at all — and run in ~0.42s mean vs. ~1.0s for the 9 that did parse.
  This looks like the model hitting immediate-EOS on ~85% of calls, a generation/stop-token
  behavior specific to `:detect` mode's exact prompt+image combination, not the same "small
  object, out-of-distribution domain" explanation given above for why the 9 parsed answers were
  themselves inaccurate. That explanation still stands for those 9 (113.5mm mean, worse than the
  centre-guess reference) — it just doesn't explain the other 51, which is a different, previously
  uncharacterized problem worth a real look (e.g. compare against a plain `generate()` call with no
  `max_new_tokens`/sampling-config changes from the smoke test, check for a truncated/mismatched
  `<image>` token count on this specific prompt shape) before assuming `:detect` mode is simply "the
  model doesn't work here." Among the 9 that did parse: black x5, red x3, yellow x1, green x0 — too
  small an n to read as a real color pattern.
- `target_label` doesn't exist in Qwen's logged JSON schema (always absent) — any future per-label
  breakdown for Qwen needs to key off `instruction` instead.

**Per-command breakdown, Qwen baseline (new, not in §8.2)** — real spread, `go_black` is the
accuracy drag, not a uniform ~25mm across all four colors:

| instruction | n | mean err (mm) | median (mm) | hit-rate |
|---|---|---|---|---|
| go_black | 13 | 41.2 | 21.4 | 0.38 |
| go_green | 18 | 27.5 | 7.5 | 0.72 |
| go_red | 16 | 20.2 | 14.3 | 0.75 |
| go_yellow | 13 | 12.6 | 8.8 | 0.85 |

The 60 sampled frames are go_red/green/yellow/black only — no directional/hold/stop frames are in
this sample, so whether the "point to the ball" fallback (§3) performs differently is still
untested.

**Session outlier**: `session_jetson_track4_20260915_151627` (the first session recorded that day)
is a real outlier at 50.95mm mean / 33% hit-rate vs. 67-85% for every other session (11.4-37.1mm
range). No per-frame lighting/environment metadata is logged, so a cold-start/warm-up cause is a
plausible guess, not confirmed.

**Error magnitude, Qwen baseline**: mostly close misses, not categorical confusion — 41/60 under
20mm (hits), 10/60 in the 20-40mm band, only 6/60 (10%) above 80mm (max 153mm). The 20mm hit
threshold is somewhat harsh given the 20-40mm "just missed" band, but a real ~10% tail of wild
misses exists too, consistent with occasional wrong-marker confusion rather than pure localization
noise.

**Latency vs. accuracy (Qwen)**: no meaningful relationship — hit-frame mean latency 1.70s vs.
miss-frame 1.70s, Pearson r=0.22 (weak). Not informative for prioritizing candidates by a
speed/accuracy tradeoff based on this data alone.

## 11. 2026-09-24: real InternVL2.5-4B results, Moondream2 hard-blocked by a torch/transformers conflict

Full sweep run on the Jetson (`arm2-t4:r36.4.0`, `jetson_run1`, `arm2_jetson_sweep_results3`).

**InternVL2.5-4B: complete, 100% parse rate, but badly inaccurate — worse than a no-vision
constant-centre guess, and gets WORSE with more prompt context:**

| variant | parse_rate | hit@20mm | mean_err_mm | median_err_mm |
|---|---|---|---|---|
| baseline | 60/60 | 3.3% (2/60) | 88.0 | 88.4 |
| oriented | 60/60 | 0% | 129.7 | 130.7 |
| oriented_aruco | 60/60 | 0% | 151.8 | 133.2 |
| *[ref] constant platform-centre guess* | *60/60* | *15% (9/60)* | *25.2* | *29.6* |

Unlike PaliGemma2 (mostly can't produce an answer at all), InternVL2.5-4B confidently produces a
parseable point on every call — it's just consistently wrong, and the `oriented_aruco` variant
(the one adding the most real platform/marker context, expected to help) is the *worst* of the
three. Not yet investigated why more context makes it worse; a real, notable finding on its own
worth a closer look if this candidate stays in scope.

**Moondream2 (`:query` and `:point`, all variants): hard-blocked, 0/180 and 0/60, not a data or
prompt problem.** Every single call fails identically:
```
TypeError: scaled_dot_product_attention() got an unexpected keyword argument 'enable_gqa'
```
Moondream2's pinned remote code (revision `2025-06-21`, `9a7d4024050840e001defacec2b00727e89149e6`)
calls PyTorch's `scaled_dot_product_attention` with an `enable_gqa` kwarg that only exists in newer
torch. `arm2-t4` is built on `dustynv/l4t-pytorch:r36.4.0` (torch 2.4.0) — kept on this older base
because Moondream2 (and InternVL2.5) need `transformers<5` (`Dockerfile.arm2-transformers4`'s own
docstring), and at the time that Dockerfile was written the newer-torch requirement was only known
to apply to `transformers>=5` (Qwen/PaliGemma2's `arm2-t5` image). This run shows that assumption
was incomplete: **Moondream2 independently needs `torch>=2.5` (for `enable_gqa`) AND
`transformers<5` at the same time** — a combination neither `arm2-t4` nor `arm2-t5` currently
provides. Not fixed tonight: a live torch upgrade inside the working `arm2-t4` container was
deliberately not attempted this session, given the real risk of silently breaking the
just-confirmed-working InternVL2.5-4B pairing (same class of risk as this project's earlier
torch/torchvision CPU-wheel-substitution incident) and being near a rate-limit boundary late at
night. **Next step for a future session**: either build a third image (`transformers<5` +
`torch>=2.5`, matching wheel provenance carefully) or confirm whether a newer Moondream2 revision
drops the `enable_gqa` call before deciding an environment change is even needed.

**Candidate coverage status after this run**: Qwen2.5-VL-3B (complete, real, best performer so
far), PaliGemma2-3b-mix-448 (complete, real, poor), InternVL2.5-4B (complete, real, poor and
context-inverted), Moondream2 (blocked, 0 real data). Per `ARM2_MINIMAL_BASELINE_SCOPE.md`'s gate,
the zero-shot baseline sweep is not yet complete enough to unpark the fine-tuning/specialization
track — Moondream2's environment gap is the one remaining blocker.
