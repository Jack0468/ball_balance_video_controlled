# Project TODO Ledger

Significant open items only — architecture decisions, real unresolved bugs, and genuine
blockers, not minor cleanup notes. `docs/PROJECT_LOGBOOK.md` holds the full history/rationale
behind every entry here; this file tracks current actionable status only, and should stay
short enough to actually be read before starting new work.

**Workflow (see `CLAUDE.md`)**: at the end of a work session, review what was attempted versus
actually completed. Anything flagged, started-but-unfinished, or explicitly deferred must be
added here before the session is considered wrapped up — don't wait to be asked.

**Format**: `- [ ] **Title** (flagged <date>, owner: <agent/person>) — one-line description.
Source: <doc/entry reference>.` Check the box and move the line to a dated "Resolved" note (or
just delete it, logbook already has the history) once genuinely done.

**Hardware-access tags** (added 2026-10-13 end-of-session pass): `[JETSON-HW]` marks an item whose
*final verification* genuinely requires the real robot (live camera/touch/serial/motors) — no
amount of code or offline-data work can close it alone. `[CLOUD]` marks an item blocked on Drive/
Colab access instead. `[HUMAN]` marks a physical-hardware or scope decision needing a person's
sign-off, not a software fix at all. An item with no tag is software/design work that can proceed
without any of these — the ledger below leads with exactly which items those are.

## Vision (`ml-vision`)

- [ ] **Vision-accuracy calibration correction built and validated, never integrated or
  hardware-tested** `[HUMAN sign-off first, then JETSON-HW]` (flagged 2026-09-24, owner: ml-vision) — a per-session affine correction
  (`host_software/ml_jetson_vla/core/vision_calibration.py`: `AffineCorrection`,
  `fit_affine_correction`, `apply_correction`, etc.) reduces mean vision error 5.6mm → 4.4-4.5mm
  (~20%) on offline-validated data, against the user's stated <3mm Steady-State Error target
  (current measured ~9-17mm). Confirmed still not wired into `run_jetson_standalone.py`
  (no reference to `vision_calibration` anywhere in that file) — the insertion point (right
  after `px_to_touch_mm()`, before `PredictionGate.filter()`) is identified but needs explicit
  user sign-off since it's a new layer on `CLAUDE.md`'s LOCKED coordinate-mapping decision.
  Also blocking: `TargetStateMachine` (confirmed, `host_software/src/state_machine.py`) still has
  no way to command an arbitrary `(x_mm, y_mm)` target, needed to drive a ball through a
  calibration grid on real hardware. Even the best validated result (~4.4mm) remains above the
  <3mm target — a second lever (control-loop/settling-dynamics tuning) is still needed and not
  yet investigated. Source: `docs/PROJECT_LOGBOOK.md` 23-24/09/2026 "Steady-State Error Root
  Cause" entry; `docs/EVALUATION_STRATEGY.md` section 1; `host_software/ml_jetson_vla/docs/VISION_CALIBRATION_PROPOSAL.md`.
- [ ] **Arm 1 vision-model-class decision still undecided: small (Shared Backbone CNN) vs.
  medium (YOLO/ResNet lineage)** `[HUMAN]` (flagged 2026-08-14, owner: user/ml-vision) — both remain live
  candidates; `CLAUDE.md`'s Architecture Decisions table states this explicitly ("don't assume
  either side") after the YOLO/ResNet ban was lifted 2026-08-13. Not a bug, a pending
  architecture decision that should be made before locking Arm 1's final pipeline. Source:
  `CLAUDE.md` Architecture Decisions table; `host_software/ml_jetson_vla/docs/ARCHITECTURE.md`
  "What's confirmed vs still open" table.
- [x] **`TargetStateMachine.maybe_auto_hold()` still reads raw, unfiltered `marker_coords`**
  (flagged 2026-09-15, fixed 2026-10-13, owner: ml-vision) — `maybe_auto_hold()` now reads
  `self.marker_filters[self.current_target_name].position` (the same jump-gated/seeded
  `_MarkerPositionFilter` estimate `get_target_coords()` already used) instead of the raw
  per-frame `marker_coords` dict, in `host_software/src/state_machine.py`. New test
  `host_software/ml_vision/tests/test_state_machine_auto_hold.py` (4 tests, all passing)
  specifically proves a noisy/misclassified raw `marker_coords` reading no longer affects
  auto-latch once the filter is confirmed. Full `host_software/ml_vision/tests/` suite re-run
  clean alongside it (same pre-existing, unrelated failures/errors as before this change — see
  `docs/PROJECT_LOGBOOK.md` 13/10/2026 entry for exact before/after counts). Source:
  `docs/PROJECT_LOGBOOK.md` 15/09/2026 "Marker Target Jump-Gate" entry, open item (1); fix logged
  13/10/2026.
- [ ] **iPhone-derived extended-eval datasets unusable for grounding eval: no language labels,
  coordinate frame mismatch** (flagged 2026-10-07, owner: ml-vision) — `02_silver_unified_pose`,
  `03_gold/images_iphone`, and the iPhone `video1-5` sets have ball/marker positions but no
  language labels, and their coordinate frame does not match `ground_truth_manifest.json` under
  either convention tested (median ~105mm off one way, a ~40-unit offset the other) — checked
  directly, not assumed. Not usable for grounding evaluation until resolved. Source:
  `docs/PROJECT_LOGBOOK.md` 07/10/2026 "Extended-eval data inventory" entry,
  `host_software/ml_jetson_vla/reports/extended_data_inventory.csv`.

- [ ] **Marker colour classification needs a robust design, not another hardcoded fix —
  REOPENED 2026-10-14, corrected framing** (flagged 2026-10-13, owner: ml-vision) — `_01`-vs-`_03`
  sheet mismatch is real and still unresolved for live operation (every HSV bin was only ever
  calibrated against `_03`; the physically-mounted sheet is `_01`). 13/10's "fix" (`MarkerClassifier`'s
  opt-in manifest-position identification tier -- matching a blob's position against the sheet
  manifest to read its colour straight off the manifest, bypassing pixel classification) was wired
  into all 4 live/operation call sites over 13-14/10, then **reverted on 2026-10-14 per explicit
  user correction**: using ground truth at inference time defeats the project's actual purpose —
  the CNN+`MarkerClassifier` pipeline is supposed to be the thing determining marker colour from
  the camera; the manifest's real role is scoring how good that determination is, and generating
  training/eval labels, never standing in for inference on the real robot. The capability itself
  (`MarkerClassifier(manifest_path=...)`) stays in `marker_classifier.py` — genuinely correct for
  evaluation/training scripts — but is no longer wired into `small_cnn_grounder.py`,
  `run_jetson_standalone.py`, `main_onnx_shared_vision_audio.py`, or
  `main_onnx_shared_vision_nemo_audio.py` (all 4 reverted, verified: 150 passed/1 pre-existing
  skip/0 regressions, a real `JetsonExpertPolicy` reconstructed and confirmed back to pixel-based
  classification). **The genuinely still-needed fix**: something that actually classifies colour
  from pixels correctly under the current sheet/lighting — per-session/per-sheet HSV
  auto-calibration, a trained/learned colour classifier, relative/contrastive colour matching, or
  explicit sheet-identity detection paired with sheet-specific *pixel* calibration loaded at
  runtime (not a position-to-ground-truth lookup). **Priority note (2026-10-14, corrected)**: this
  blocks trusting any live colour-grounding result from the policies that actually route through
  `MarkerClassifier` — `hybrid_small_cnn_act` (`SmallCnnGroundingPolicy`) and the original
  production scripts (`run_jetson_standalone.py`, both `main_onnx_shared_vision_*.py`). It does
  NOT affect `hybrid_qwen_act` or `smolvla_direct` — checked directly, not assumed: Qwen grounds
  itself via its own VLM reasoning directly on the image (`MinimalVLMPolicy`/`QwenVLBackend`,
  never touches `MarkerClassifier`), and `SmolVLADirectPolicy` grounds itself on the instruction
  text alone (no `_Grounder`, no marker classifier reference anywhere in its construction). Source:
  `docs/PROJECT_LOGBOOK.md` 13/10/2026
  entries (`f88877e`, `9c76748`, manifest-tier shipped) and 14/10/2026, "Reverted the
  manifest-position colour-ID fix out of all 4 live/operation call sites."

## Audio (`ml-audio`)

- [ ] **Unbounded live audio chunk queue with blocking inference — no backpressure or staleness
  detection** (flagged 2026-10-14, owner: ml-audio) — `nemo_live_receiver.py`'s
  `self.chunk_queue = queue.Queue()` has no `maxsize`; `audio_receiver_pytorch.py` has the
  identical shape. A single consumer thread runs a blocking model forward pass before pulling
  the next chunk; if that ever takes longer on average than `step_seconds` (0.2s default) —
  plausible on CPU, especially sharing the Jetson with vision inference — chunks silently
  accumulate with nothing bounding the queue, dropping stale chunks, or even logging it.
  `latest_inference_time_ms` is computed but only ever read for cosmetic display, never compared
  against `step_seconds`. Concrete failure: the audio command stream progressively lags real
  time, unboundedly, with the fusion state machine potentially acting on a "live" command that's
  actually seconds stale during real robot operation — exactly the class of bug this project's
  own non-blocking-execution constraint (`CLAUDE.md`) exists to prevent. Not yet measured live;
  the code has no mechanism to prevent or detect it either way. Fix: bound the queue with a
  drop-oldest policy, or a watchdog comparing `latest_inference_time_ms` against `step_seconds`.
  Source: `host_software/ml_audio/evaluations/nemo_live_receiver.py` lines 100-105, 187-224;
  `host_software/ml_audio/audio_receiver_pytorch.py` lines 147, 234-274.
- [ ] **Module-level mutable `LABEL_NAMES` global in the legacy PyTorch receiver — same bug
  class that already cost ~80 accuracy points in production** (flagged 2026-10-14, owner:
  ml-audio) — `audio_receiver_pytorch.py`'s `__init__` does `global LABEL_NAMES` and reassigns
  it; `_process_loop` decodes predictions from that same global rather than an instance
  attribute (the NeMo receiver already does this correctly via `self.labels`). Constructing two
  `AudioCommandReceiver` instances against checkpoints with different class counts/orders in one
  process would have the second construction silently overwrite the label mapping used by the
  first instance's already-running thread — the identical failure shape as the documented
  production incident where wrong label order crashed accuracy from ~86% to ~7.4%
  (`audio_eval_notebook_refactor_plan.md`), just reachable through a different trigger. Current
  blast radius is low (nothing in the repo today constructs two receivers in one process), but
  given this exact bug class already happened once, worth closing. Fix is trivial: `self.labels`
  instead of the global, everywhere. Source: `host_software/ml_audio/audio_receiver_pytorch.py`
  lines 30-42, 82, 104/106/116, 258.
- [ ] **Production entry point's default audio checkpoint is still the preliminary, never
  live-stream-scored one — contradicts `CLAUDE.md`'s own "Current State" narrative** (flagged
  2026-10-14, owner: ml-audio) — `main_onnx_shared_vision_nemo_audio.py`'s `DEFAULT_NEMO_MODEL`
  points at `nemo_matchboxnet_3x1x64_noisemix_seed0` (the code's own comment already calls this
  "Preliminary/half-trained"), while `CLAUDE.md` describes `nemo_matchboxnet_v1` (8-9/10
  live-stream, two independent runs) as the established result. Per the plan doc, the seed0
  checkpoint never finished its original training run and its live-stream accuracy against the
  real receiver harness has never been measured at all -- flagged as intentional for
  integration-testing on 2026-09-15/16, but nothing since shows it was ever switched back, and
  it's now 2026-10-14. Anyone running the production script with no flags gets the
  unvalidated checkpoint; `--audio-model` must be passed explicitly to get the actually-
  recommended one. Fix is trivial (flip the default) once someone confirms via
  `evaluate_nemo_live_receiver_stream.py` whether the seed0 checkpoint is even worth keeping as a
  reference point. Source: `host_software/main_onnx_shared_vision_nemo_audio.py` lines 97-102,
  128.
- [ ] **No automated regression coverage of the actual production audio inference/preprocessing
  path** (flagged 2026-10-14, owner: ml-audio) — `ml_audio/tests/test_audio.py` opens a mic
  stream for 2 seconds and prints whether it errored; zero assertions, not wired to any
  classifier/DSP code. `test_nemo_noise_mix.py` is a real smoke test but only exercises
  training-time noise-perturbation wiring (needs a separate conda env), not a regression guard
  on `_process_loop`/`audio_dsp.py`. This project has already paid for two severe
  train/production-preprocessing-divergence bugs (wrong label order; mismatched crop/
  renorm/noise-profile), both only caught by manual A/B investigation -- nothing today would
  fail if either regressed again. Real fix (moderate effort, worth scoping separately): assert
  the offline eval path and the live `_process_loop` path produce numerically identical features
  for the same input clip, plus a fixed-checkpoint label-order sanity check. Source:
  `host_software/ml_audio/tests/test_audio.py`; `audio_eval_notebook_refactor_plan.md`
  "Evaluation Extraction" section.

## Shared / Fusion (`host_software/src/`)

- [ ] **A real, producible audio command ("go_grey") has no corresponding marker filter and
  silently degrades to "go to center" forever** (flagged 2026-10-14) — `state_machine.py`'s
  `valid_targets` includes `"grey"`, but `self.marker_filters` only ever has `blue/green/red/
  yellow/black` -- no `"grey"` entry, ever. This is a real, reachable case, not theoretical:
  `audio_receiver_onnx.py`'s `LABEL_NAMES` confirms `"go_grey"` is a genuine output class of the
  production NeMo model, forwarded unmodified into `state_machine.process_command()`, and
  `CLAUDE.md`'s own Current State table lists "go_green/go_grey confusion" as a live current
  audio failure mode -- i.e. the model does predict this label in practice. The live
  `MarkerClassifier` only ever emits `blue/red/yellow/green/black` (no grey class at all;
  `marker_tracker.py` has a grey HSV bin but that module is confirmed off the live inference
  path, per an already-closed TODO item). Concrete failure: when "go grey" is correctly
  recognized by audio, `_apply_command()` accepts it (valid_targets allows it), but every
  `get_target_coords()` call then does `marker_filters.get("grey")` -> `None` -> falls through to
  the "not yet confirmed" branch -> returns `(0.0, 0.0)` (center), permanently -- indistinguishable
  at runtime from a legitimate "still confirming" transient, except it never resolves and nothing
  logs or errors. Fix options (not an agent's call to pick unilaterally): reconcile
  `valid_targets` against the real 5-class marker taxonomy, add an explicit reject/log path for a
  colour with no filter, or give vision a real grey class. Source:
  `host_software/src/state_machine.py` lines 140-154, 164-170;
  `host_software/src/audio_receiver_onnx.py` lines 26-39.

## Multimodal / VLA (`ml-multimodal`)

- [ ] **ACT_FAST retrain on the safety-filtered dataset — data side fully done, actual Colab
  retrain not yet run** `[CLOUD to retrain, then JETSON-HW to re-test the new checkpoint live]`
  (flagged 2026-10-13, updated same day) — filter implemented in
  `convert_to_lerobot.py`, committed, tested. Real measured per-frame counts: `pid_webcam` 45,840
  kept / 65,153 dropped of 110,993 (41.30%/58.70%); `jetson_track4_rl` all 40,713 rows untouched.
  **Full LeRobotDataset rebuild now complete and verified** (first attempt died ~100min in,
  workspace-move suspected; retry succeeded): real output confirmed `total_episodes: 15`,
  `total_frames: 86553` (exactly 40,713+45,840). `verify_and_package_dataset.py
  --check-video-decode` confirms every one of 86,553 frames across all 8 video files decodes
  cleanly, zero corruption. Packaged: `lerobot_full_2026_10_13_safety_filtered.zip` (~1.13 GiB,
  sha256 `3e7014eb...`), currently sitting in this session's local scratch path (not yet on
  Drive). `act_train_colab.ipynb` already points at `run_003`/this exact zip filename. **Still
  outstanding, needs the user's own Drive/Colab access**: upload the zip to
  `MyDrive/2026_VRI_WINTER/colab_bundle/`, then actually run the notebook to retrain, then
  re-test live on hardware. Source: `docs/PROJECT_LOGBOOK.md` 13/10/2026, "Safety-filtered LeRobot
  dataset rebuild completed, verified, packaged."
- [ ] **Eval cell (h) torchcodec crash in `act_train_colab.ipynb`** `[CLOUD]` (flagged 2026-10-13, patched
  same day — NOT yet confirmed working) — diagnosed as likely a Colab-side decoder-backend issue
  (torchcodec vs. the software decoder used to verify the dataset's own video files are clean),
  not source-data corruption. Cell (h) now monkeypatches `torchcodec.decoders.VideoDecoder` to
  force `device="cpu"` and reports the exact batch/episode range on any decode failure instead of
  a plain `for batch in loader:` (notebook JSON verified to still parse, new cell source verified
  to compile as valid Python). **Structural fix pending the user's next real Colab run** — no
  Colab/GPU access from this session, so whether the crash actually disappears is still unknown.
  If it still crashes, remove the patch and reopen the investigation using the newly-named
  batch/episode. Source: `docs/PROJECT_LOGBOOK.md` 13/10/2026, "`act_train_colab.ipynb` cell (h)
  patched to force torchcodec onto CPU decode" and "Eval-only torchcodec video-decode crash."
- [ ] **Qwen grounding full Colab run still interim** `[CLOUD]` (flagged 2026-10-13) — only 5 of 10
  sessions scored as of last check (93.3% ball-xy hit@20mm, 59.9% marker-grounding -- both
  interim). Re-score once all 10 sessions finish; the marker-grounding discrepancy vs. the
  validated 68.3% sweep figure is unexplained and needs the full run to resolve. Source:
  `docs/PROJECT_LOGBOOK.md` 13/10/2026, "Real interim Qwen2.5-VL-3B grounding results."
- [x] **Live hardware test of a real trained ACT checkpoint (checklist item 9)** (flagged
  2026-10-13, resolved same day) — confirmed live: 205 lines sent, 3/3 chunks accepted, real
  touch+grounding. Root cause of 3 earlier failed attempts (looked like a hang, was actually a
  silently-swallowed chunk-size-mismatch crash) fixed in `fast_layers.py`/`core/hybrid_policy.py`.
  Source: `docs/PROJECT_LOGBOOK.md` 13/10/2026, "Checklist item 9... root-caused a 3-run hang."
- [ ] **Touch-sensor single-position anomaly, unexplained** `[JETSON-HW]` (flagged 2026-10-13) — one specific
  ball placement (overlapping an ArUco marker directly) produced 0/136 valid touch samples,
  while all four platform corners tested normal read fine. Localized, not systemic, not chased
  further. Source: `docs/PROJECT_LOGBOOK.md` 13/10/2026, H1 corner-testing entries.
- [x] **No periodic re-grounding: colour targets go stale for seconds at a time** (flagged
  2026-10-06, resolved 2026-10-13, owner: ml-multimodal) — `core/hybrid_policy.py`'s `act()` now
  re-submits a held colour command to the grounder on a cadence (`DEFAULT_REGROUND_INTERVAL_S`,
  1.0s default, constructor-overridable) instead of grounding once on instruction-change and
  never again, bounding steady-state target age by roughly `max(interval, grounding backend
  latency)` instead of letting it grow for the whole hold duration (previously measured p90
  ~4.2-8.9s, well past the project's own proposed <=2000ms criterion). Verified with a new
  synthetic test (`tests/test_hybrid_policy_cpu.py`'s `PeriodicRegroundingTests`, driven through
  the real `_Grounder`/`_FastWorker` threads) showing measured target age stays under 2000ms
  while a command is held, and that `SmolVLADirectPolicy` is structurally unaffected. Full
  `ml_jetson_vla/tests/` suite: 133 passed/1 skip before, 136 passed/1 skip after, 0 regressions.
  **Not yet re-measured against a real session replay/live run with the real Qwen or small-CNN
  backend** (needs Jetson/Docker access) — the old 4.2-8.9s figures are the pre-fix baseline this
  closes, not superseded by a new measured number yet. Source: `docs/PROJECT_LOGBOOK.md`
  2026-10-06/07/09/13 entries (13/10's "Periodic re-grounding added..." entry has the full
  writeup); `host_software/ml_jetson_vla/docs/EXPERIMENT_OPTIONS_PLAN_2026_10_07.md` section 4,
  "Known limit #2" (updated in place).
- [ ] **SmolVLA's 6-to-3 output reduction and state zero-padding are undecided placeholders**
  `[HUMAN]` (flagged 2026-10-07, owner: ml-multimodal) — `SmolVLADirectPolicy`/`SmolVLAFastLayer` reduce
  the model's native 6-value action output to this platform's 3 (`theta_a/b/c`) by taking the
  first 3 values, and zero-pad the 2-dim real state to the checkpoint's native state width; both
  are explicitly flagged as placeholders, and the output-reduction scheme in particular is an
  "open decision [that] must be fixed before fine-tuning" — never decided. Source:
  `host_software/ml_jetson_vla/docs/EXPERIMENT_OPTIONS_PLAN_2026_10_07.md` section 2 and
  section 6.
- [ ] **Hybrid Qwen+ACT pipeline's real grounding accuracy under `arm2-lerobot`'s transformers
  4.57.6 has never been scored against ground truth** `[JETSON-HW]` (flagged 2026-10-09, re-confirmed still
  open 2026-10-13, owner: ml-multimodal) — `bench_hybrid_qwen_act.py` gained the scoring
  instrumentation (`qwen.grounding_accuracy`) 2026-10-09, but no real hybrid run's output has
  ever been saved/scored with it. **2026-10-13: checked both existing saved outputs that could
  plausibly have this** (`reports/arm2_action_latency/hybrid_qwen_act.json`,
  `reports/policy_replay_hybrid_act_real.json`) **and a repo-wide grep for `grounding_accuracy`
  — confirmed neither has it and nothing else does either.** The saved `hybrid_qwen_act.json` run
  predates the instrumentation (created 02:55 UTC 09/10, instrumentation added later that day);
  `policy_replay_hybrid_act_real.json` comes from a different driver (`run_policy_replay.py` /
  `core/hybrid_policy.py`) that was never wired to the scorer at all. **No number was fabricated
  and no Jetson/Docker run was attempted (not available from this session).** This matters
  because transformers 4.57.6 (the `arm2-lerobot` image's pin, needed to co-locate Qwen+ACT on
  one GPU) was separately confirmed DIVERGENT from transformers 5.17.0 (the sweep's validated
  environment, 68.3% hit@20mm): 22/60 exact match, mean pixel diff 24.5px, one frame off by
  256px, one unparseable. Any hybrid-pipeline run's grounding accuracy should not be assumed to
  match the validated 68.3% figure until actually scored. **Exact next step to close this**: run
  `bench_hybrid_qwen_act.py` for real inside `arm2-lerobot:r36.4.0` on the Jetson and commit its
  output JSON to `host_software/ml_jetson_vla/reports/` — no further code change is needed, the
  current script already populates `qwen.grounding_accuracy` on every run. Source:
  `docs/PROJECT_LOGBOOK.md` 09/10/2026 "Qwen transformers 4.57.6 vs 5.17.0 parity" and "Hybrid
  benchmark can now measure its own real Qwen grounding accuracy" entries; 13/10/2026 "Checked
  for a saved, scorable `hybrid_qwen_act` run before trying to produce one" entry.
- [ ] **Task Success Rate can't distinguish a dropped ball from a trial that never settled**
  `[design is software-only; confirming a chosen signal against real drops is JETSON-HW]`
  (flagged 2026-09-18, owner: ml-multimodal/orchestration) — confirmed still true in
  `evaluate_system_control.py` (its own comment: "no ball-drop signal exists in current
  telemetry, so a dropped ball is NOT distinguished from a trial that simply never settled").
  Closing this needs an upstream drop-detection signal (e.g. from vision), not a change to the
  evaluator itself. Source: `docs/EVALUATION_STRATEGY.md` "Evaluation Pipeline" section;
  `host_software/evaluations/evaluate_system_control.py`.
- [x] **Ball-trajectory jerk/smoothness metric** (flagged 2026-09-18/2026-09-24, implemented
  2026-10-13, owner: ml-multimodal) — **the old blanket "needs new instrumentation" framing did
  not actually hold for this metric once checked concretely; it was correct only for its sibling
  below.** `touch_x`/`touch_y` + a per-frame timestamp are already in every telemetry schema
  `evaluate_system_control.py` accepts, which is all a 3rd-time-derivative-of-position metric
  needs. Implemented as a standalone module, `host_software/ml_multimodal/evaluations/
  trajectory_smoothness.py` (`jerk_metrics_for_segments()`), NOT inside `evaluate_system_control.py`
  itself — that file is outside `ml-multimodal`'s write boundary
  (`host_software/evaluations/` is read-only for this agent). 10 new tests
  (`ml_multimodal/tests/test_trajectory_smoothness.py`, all passing), including an analytic
  cubic-trajectory check against a known constant jerk, degrade-gracefully cases (too-short
  segment, non-monotonic timestamps), and a **real verification against
  `data/01_bronze/session_jetson_track4_20260915_151627/telemetry.csv`** (11 trial segments, 0
  excluded): `Average_Jerk_Mean_mm_s3 ≈ 16361`, `Average_Jerk_RMS_mm_s3 ≈ 45838` unfiltered,
  dropping to `≈10622`/`≈14002` with `--filter-touch-outliers` (confirms real sensitivity to the
  known touch-plate glitch pattern — always report alongside whether filtering was applied).
  **Wired into `evaluate_system_control.py` 2026-10-13 (end-of-session pass)**:
  `compute_metrics()` now merges `**jerk_metrics_for_segments(times, touch_x, touch_y, segments)`
  into its return dict; re-run end-to-end against the same real telemetry gave identical numbers
  to the standalone verification above, confirming correct wiring. Fully closed. Source:
  `docs/EVALUATION_STRATEGY.md` "Extended Metrics" section (updated in place);
  `docs/PROJECT_LOGBOOK.md` 18/09/2026 "Telemetry
  Field-History Audit" entry; 13/10/2026 "Ball-trajectory jerk/smoothness" entry.
- [ ] **Disturbance-rejection/robustness metric under injected perturbation never implemented**
  `[JETSON-HW — needs a real physical perturbation test protocol, not a formula]`
  (flagged 2026-09-18, re-confirmed still open and genuinely instrumentation-blocked 2026-10-13,
  owner: ml-multimodal/orchestration) — confirmed still absent from `evaluate_system_control.py`.
  Unlike jerk/smoothness (above, now implemented), this one genuinely needs a dedicated hardware
  perturbation test protocol (injecting a real disturbance and measuring recovery), not a new
  formula over existing telemetry — flagged twice (2026-09-18, 2026-09-24) and still not built
  either time, deliberately left alone this session per that same scoping. Source:
  `docs/EVALUATION_STRATEGY.md` "Extended Metrics" section; `docs/PROJECT_LOGBOOK.md` 18/09/2026
  "Telemetry Field-History Audit" entry.
- [ ] **Track 2 and Track 3 (Phase 6 workstreams) have never been named or scoped** `[HUMAN]` (flagged
  2026-08-14, owner: orchestration) — `CLAUDE.md`'s own Terminology section states this
  plainly: only Track 1 (small-class expert pipeline) and Track 4 (in-house large-VLA research
  spike) are named/scoped; "Track 2 and 3 have never been named or scoped in any doc — a real,
  open gap, not an omission to paper over." Source: `CLAUDE.md` "Terminology: Phase / Track /
  Arm" section.
- [ ] **`VLADataset`'s vocabulary never matches real audio-command strings — language
  conditioning is a no-op for every real sample** (flagged 2026-10-14, owner: ml-multimodal) —
  `core/dataset.py`'s `self.vocab` uses space-separated keys (`"go red"`, `"go blue"`, ...); the
  real recognized vocabulary (confirmed directly against 10 real `session_jetson_track4_*`
  telemetry CSVs) is underscore-separated (`go_red`, `go_green`, `go_yellow`, `go_black`, `hold`,
  `forward`/`backward`/`left`/`right`/`stop`) and matches `minimal_vlm_policy.py`'s
  `COLOR_COMMANDS` exactly. None of the real strings hit a vocab key except `"hold"` — every
  single real-language-labeled sample (~40,713 `jetson_track4_rl` rows) silently resolves to
  `cmd_idx=0`, identical to "hold"; `"go_black"` has no vocab entry in any spelling. Confirmed
  directly: `self.vocab.get(item.get("audio_command","hold"), 0)` against the real dict literal.
  Since `cmd_idx` is the *only* language input (`FiLMLayer` modulation over the ResNet feature
  map in `vla_architecture.py`), `RT1LiteVLA` as actually trained by `train_vla.py` has never
  been language-conditioned at all despite the architecture being built for it. No test exists
  for this mapping. Fix is trivial (normalize/extend the vocab to match real strings), but any
  existing `models/vla_v1/best_vla.pth` checkpoint needs retraining under the corrected mapping.
  Source: `host_software/ml_multimodal/core/dataset.py` lines 14-20, 40;
  `host_software/ml_multimodal/core/vla_architecture.py` lines 67-75.
- [ ] **`run_eval_baseline_vla.py`/`run_eval_our_vla.py` are 100% mocked but write
  schema-correct output indistinguishable from a real Arm comparison** (flagged 2026-10-14,
  owner: ml-multimodal) — both scripts never touch a camera or the firmware: mock marker
  coordinates, mock touch state (`tx+8.0, ty+2.0`), a zero image tensor. `run_eval_our_vla.py`
  comments out the one line that would load different weights than the baseline, so both
  literally run the same untrained, randomly-initialized model. `rl_finetuned_vla.pth` (what
  "our VLA" is supposed to load) doesn't exist anywhere — `train_vla.py`'s `fine_tune_rl()` only
  prints status text, no real RL implementation. Both write to the exact column schema
  `evaluate_system_control.py` expects, so running them and feeding the output into a real
  cross-arm comparison produces a completely fabricated, internally-plausible-looking 4-metric
  result with zero real inference/actuation behind either side — a live landmine given this
  project's current top priority is exactly this kind of comparison. Cheap mitigation available
  now: rename/flag the outputs so they can't be mistaken for real telemetry, and make
  `fine_tune_rl()` raise `NotImplementedError` instead of printing success-sounding text; the
  real fix (actual camera/serial integration + a real Stage-2 RL implementation) is substantial.
  Source: `host_software/ml_multimodal/evaluations/run_eval_baseline_vla.py` lines 17-18, 65-75,
  100-112; `run_eval_our_vla.py` lines 12-13; `train_vla.py` lines 112-129.
- [ ] **`VLADataset.__getitem__`'s bare `except:` silently substitutes a blank image with zero
  logging** (flagged 2026-10-14, owner: ml-multimodal) — catches everything (including
  `KeyboardInterrupt`/`SystemExit`), substitutes a black placeholder image, no counter/log of how
  many samples were blanked in a run. Low risk today since `generate_vla_dataset.py` writes a
  real file per sample at build time, but a real risk against the kind of path-breakage
  `docs/DATA_STORAGE.md`'s own portability audit flags (drive remount, a refactor moving
  `03_gold/images/`, a partial sync) — training/eval would silently proceed on blanked samples
  with no visible symptom beyond a slightly worse loss curve. Fix is trivial: narrow to
  `(FileNotFoundError, OSError)`, count/log fallbacks, consider raising past a threshold. Source:
  `host_software/ml_multimodal/core/dataset.py` lines 30-34.
- [ ] **Jetson expert-pipeline control-net port (`core/control_net.py`) is further along than
  `CLAUDE.md`'s current framing suggests, and its velocity-filter timing is unverified against
  real achieved loop rate** `[the dt-comparison analysis itself needs no hardware -- existing
  telemetry has what's needed; trusting the result on real motors is JETSON-HW]` (flagged
  2026-10-14, owner: ml-multimodal) — this module is a real, working port already wired into
  `runtime/run_jetson_standalone.py --remote-control` (Phase B), with matching firmware already
  built and functioning at `firmware/stm32_jetson_remote_control/BallBalancingBot/
  RemoteStepControl.cpp` (confirmed: parses `S,<a>,<b>,<c>`, clamps to +/-98 steps, fails safe to
  level after 3s stale). Worth noting since the project's own current-state framing describes
  this track as earlier-stage than the code actually is. Real open risk: `ControlNet.infer()`'s
  `VEL_FILTER_ALPHA=0.35` EMA (mirroring `RLControl.cpp`) is applied identically regardless of
  `actual_dt`, but the Jetson's camera-frame-bound loop (30fps hard cap + ArUco + CNN + marker
  overhead) almost certainly runs at a looser, less uniform dt than the STM32's native loop that
  generated the as-deployed training distribution -- a fixed-alpha EMA's effective time constant
  is dt-dependent, so the filtered velocity could lag more than the policy was ever exposed to.
  Cheap first step, no hardware needed: compare the real dt histogram from existing
  `session_jetson_track4_*/telemetry.csv` (STM32-native) against a dry-run of the Jetson loop's
  achieved frame timing, before trusting Phase B output on real hardware. Source:
  `host_software/ml_jetson_vla/core/control_net.py` (`VEL_FILTER_ALPHA`, line ~152-153);
  `firmware/stm32_jetson_remote_control/BallBalancingBot/RemoteStepControl.cpp`.

## FPGA

- [ ] **Arm 3 framing decision** `[HUMAN]` (flagged 2026-10-13) — is Arm 3 a real fabricated photonic
  chip, a simulation/co-design-only arm, or a commercial product? Everything in the first-draft
  interface spec is written conditional on "real chip" since that's the only scenario with
  concrete numbers available -- this is a one-line question that unblocks or reframes the whole
  spec, not yet answered. Source: `fpga/docs/optical_bridge_interface_spec_draft.md` section 1.
- [ ] **Camera/UDP video streaming** `[FPGA/camera hardware — not the Jetson robot, but still
  real physical hardware to verify against, not software-closeable alone]` (status per
  `CLAUDE.md`'s Current State table: "separately unresolved") -- not touched this session,
  carried over from project state.
- [ ] **`stepper_motor_controller.v`'s step-rate constants are calibrated for 48MHz; porting to
  the Zynq fabric clock (100MHz) silently changes real motor behaviour** `[needs synthesis/
  hardware timing check, not Jetson]` (flagged 2026-10-14, owner: fpga) — `p_term = abs_error <<
  6` and the `700` max-speed clamp ("~2kHz step rate max") are tuned against a 24-bit phase
  accumulator validated specifically at 48MHz (confirmed in `fpga/tests/motor_test/test_motor.py`'s
  `pll.SetReference(48.0)` and `single_motor_test_top.v`'s "PLL0 48MHz"). This project's own
  `zynq_camera_sys` block design (`camera_system.hwh`) sets `PCW_FPGA0_PERIPHERAL_FREQMHZ = 100` —
  if Domain 4 ports this module into that same block design (the precedent its own doc comment
  points to) without re-deriving these constants, the real max step rate silently becomes
  ~4.17kHz instead of ~2kHz and the whole proportional-gain curve shifts, with no compile/sim
  error to catch it (same class of clocking assumption this project has already been burned by
  once). Fix: re-derive `<<6`/`700` for the real target clock, or parameterize on `CLK_FREQ_HZ`.
  Source: `fpga/main_controller/stepper_motor_controller.v` lines 19-23.
- [ ] **`PIDControllers.cpp`'s hardcoded `dt=0.001` (1000Hz) contradicts `CLAUDE.md`'s own binding
  120Hz/8.3ms loop-rate constraint** `[needs a timing-architecture decision, not just a constant
  edit]` (flagged 2026-10-14, owner: fpga) — `integ_x += error_x * dt` /
  `deriv_x = (error_x - error_prev_x) / dt` both use this constant directly; at a real ~8.3ms call
  interval the integral term under-accumulates by ~8.3x and the derivative term is over-amplified
  by ~8.3x relative to whatever the gains (`kp=0.8, ki=0.2, kd=0.09`) were tuned against.
  `balance_controller` is `s_axilite`-controlled (PS-polled), and nothing currently guarantees a
  deterministic call interval -- the real fix needs that invocation mechanism (timer-triggered vs.
  software-scheduled polling) decided first, not just a new constant. Source:
  `fpga/hls_hardware/PIDControllers.cpp` lines 13, 41, 44; `CLAUDE.md` System Constraints.
- [ ] **No reset path for `balance_controller`'s persistent integrator state, and `ik_solver`
  fails silently with no fault signal** `[needs Vitis HLS semantics check + a design decision]`
  (flagged 2026-10-14, owner: fpga) — `error_prev_x/y`, `integ_x/y` are `static` C variables
  persisting across every AXI-Lite-triggered call with no exposed port to clear them (e.g. on
  ball-drop/re-placement -- plausible in practice given ball-drop detection doesn't exist yet,
  see the Task-Success-Rate item above); genuinely unconfirmed whether Vitis HLS 2025.2's default
  `ap_rst` resets static C state or only control-FSM state -- needs checking, not assuming either
  way. Separately: `ik_solver` (`InverseKinematics.cpp` lines 164-167) silently returns
  `{alpha:0, theta2:0}` on an unreachable target, propagated straight to `out_thetaA/B/C` with no
  fault bit anywhere on the interface -- one leg going unreachable would silently command that
  motor to 0° while the other two keep running, a real asymmetric-platform-motion risk once wired
  to real motors. Source: `fpga/hls_hardware/PIDControllers.cpp` lines 16-19;
  `fpga/hls_hardware/InverseKinematics.cpp` lines 164-167.
- [ ] **`calculate_triangle`'s tilt model may only rotate the Z-coordinate of each attachment
  point, not the full rigid-body position -- unverified, exactly what the planned testbench
  should target** `[needs the Domain-4 digital testbench, at/near max tilt specifically]` (flagged
  2026-10-14, owner: fpga) — the loop (`InverseKinematics.cpp` lines 122-128) explicitly comments
  "keep x/y unchanged" and only computes rotated Z. At the system's own `max_angle=12.5°` clamp,
  the lateral displacement this would ignore is `~86.2mm × sin(12.5°) ≈ 18.6mm` against an 86.2mm
  arm radius -- roughly 21%, not obviously negligible. Not asserted as a bug (could be a
  deliberate, correct simplification for this delta robot's real joint geometry) -- flagged
  because it's unverified and would be invisible at small tilt angles, so the first testbench
  case should specifically exercise near-max tilt, not just small angles. Source:
  `fpga/hls_hardware/InverseKinematics.cpp` lines 122-128; `PIDControllers.cpp` line 10
  (`max_angle`).
- [ ] **Repeated, unshared transcendental HLS math calls — real but unverified risk against the
  220-DSP ceiling** `[needs a real `csynth` run, not assumable from source alone]` (flagged
  2026-10-14, owner: fpga) — `get_angles()` calls `ik_solver` three times (once per leg); each
  call plus `calculate_triangle` pulls in `hls::sin/cos/sqrt(x2)/acos/atan2` on
  `ap_fixed<32,16>`, with no `#pragma HLS ALLOCATION` or other resource-sharing directive anywhere
  in either file -- absent automatic tool inference, this likely synthesizes 3 separate hardware
  instances of each transcendental op rather than time-multiplexing one. Whether that fits inside
  220 DSP slices is genuinely unknown without running synthesis (not asserted either way).
  Separately unconfirmed: whether `hls::sin/cos/sqrt/acos/atan2` in Vitis HLS 2025.2's
  `hls_math.h` accept `ap_fixed<32,16>` directly as written -- check against a real `csynth` run
  or current UG1399 docs before assuming it compiles as-is. Note: `fpga/external_ip/
  CORDIC-Unit-main/` already has a working CORDIC IP core with its own testbenches, unused here --
  worth a look if `hls_math.h`'s built-in trig proves too resource-hungry. Source:
  `fpga/hls_hardware/InverseKinematics.cpp` lines 181-210.

## Hardware (human/EE decisions — not an agent's call)

- [ ] **Power connector, FPGA power delivery, microphone integration, PCB consolidation** `[HUMAN]` — all
  four listed as open in `CLAUDE.md`'s "Hardware Decisions (OPEN)" table, requiring human/EE
  sign-off. No agent should choose or design around these without that sign-off.
- [ ] **Touch-plate sensor transient-spike artifact's root cause still unconfirmed at the
  circuit/ADC level** `[JETSON-HW — raw ADC logging code can be drafted without hardware, but
  means nothing until run/probed on the real touch plate]`
  (flagged 2026-09-18, owner: Electrical Engineer/firmware owner) — isolated
  single-frame jumps in `touch_x`/`touch_y` (usually positive, landing in a recurring
  ~40-98mm band, reverting on the very next sample) are real and quantified (`touch_stale`/
  `touch_glitch` detectors exist and flag them), but the underlying cause is still only a
  hypothesis: `TouchProbe.cpp`'s 2-of-3-sample majority vote and its dynamic pin
  drive/read-role reconfiguration every scan are plausible glitch sources, unconfirmed —
  confirmed still true, no raw-ADC logging has been added to `TouchProbe.cpp` to investigate
  directly. Needs hardware-level investigation (raw ADC logging or direct probing), not further
  data analysis alone. Source: `docs/PROJECT_LOGBOOK.md` 18/09/2026 "Touch-Plate Ground-Truth
  Sensor Artifact" entry; `firmware/stm32_ml_control_and_vision/BallBalancingBot/TouchProbe.cpp`.
- [ ] **`TouchProbe.cpp`'s `SCREEN_HEIGHT_MM` hardcoded to 141.0, not the confirmed 142.0mm
  platform height** (flagged 2026-10-14) — directly scales every `touch_y_mm` value this
  firmware streams to the host. `CLAUDE.md`'s System Constraints table states Height=142.0mm is
  the source of truth ("never use old values"), and `docs/PROJECT_LOGBOOK.md`'s "CAD Plate vs.
  Resistive Touchpad Dimensions" entry independently confirms the resistive touchpad itself is
  187.5x142.0mm. `141.0` has been the value since this file's very first commit, never changed.
  A real, systematic ~0.7% scale bias on every logged/streamed `touch_y` reading -- already
  *noticed* in a code comment (`touch_ground_truth_filter.py` lines 84-90, which explicitly calls
  it "a separate, pre-existing firmware/spec discrepancy, not something this filter corrects")
  but never promoted to this ledger or fixed in firmware. Distinct from the transient-spike item
  above (that's single-frame glitches; this is a constant calibration bias). Fix is a trivial
  one-line constant change, but re-validating historically-collected telemetry against the
  corrected scale and re-running this same file's own "FRAME MATCH" reconciliation procedure
  (lines 41-63, which already flags a second, larger, already-known vision/touch frame mismatch)
  is non-trivial -- not something to do casually without re-running that whole calibration pass.
  Confirmed directly: `grep SCREEN_HEIGHT_MM` shows `141.0` at line 35. Source:
  `firmware/stm32_ml_control_and_vision/BallBalancingBot/TouchProbe.cpp` lines 35, 166.
- [ ] **Two complete, never-invoked control subsystems compile into the confirmed-live firmware
  image itself; a 7th `BallBalancingBot`-bearing directory is missing from `CLAUDE.md`'s own
  duplication inventory** (flagged 2026-10-14, owner: whoever owns firmware orchestration — not
  a single-lane agent's call) — `BallBalancingBot.ino`'s `setup()`/`loop()` only ever reach
  `RLControl.cpp`'s direct step-space control; confirmed never called from either function, yet
  compiled into the same binary via Arduino's whole-folder compilation: (1) `InverseKinematics.cpp/.h`
  + `MotorControl.cpp`'s `move_to_angle()` (the live path bypasses IK entirely, commanding
  steppers directly in step space); (2) `RemoteStepControl.cpp/.h` — a complete second serial
  parser/control loop (`"S,<a>,<b>,<c>"` protocol), structurally identical in purpose to what
  `firmware/stm32_jetson_remote_control/BallBalancingBot/` actually uses, but never invoked here.
  Minor supporting evidence of the same pattern: `weights_old1.h` ships alongside the actually-
  included `weights_best_so_far1.h`; `MotorControl.h` declares `test_motor_speed()` with no
  definition or call site anywhere. Why it matters: this is the exact "which code path is live"
  confusion `CLAUDE.md`'s own firmware-duplication-risk note warns about across *directories*,
  found instead *inside* the one directory that note names as unambiguously current -- a future
  edit to this dead code, or an attempt to port a Jetson-side `RemoteStepControl.cpp` fix in
  here, would silently touch nothing live. **Separately**: `firmware/stm32_jetson_vla_angle_control/
  BallBalancingBot/` (with its own `AngleStepControl.cpp`, confirmed real and live-hardware-
  tested per the 13/10/2026 H1/H2 entries) is a 7th `BallBalancingBot`-bearing directory
  `CLAUDE.md` never mentions at all -- not in the "current source of truth" line, not in the
  "intentional separate deployment" callout, not in the six-directory duplication list. Not
  deep-audited (out of the reporting agent's scope); flagged because its complete absence from
  `CLAUDE.md`'s own inventory, despite real recent hardware-validation history, looks like a gap
  in that inventory rather than evidence it's stale. Fix is trivial to delete/relocate the dead
  code once intentionality is confirmed with firmware's owner -- not something to do
  unilaterally. Source: `firmware/stm32_ml_control_and_vision/BallBalancingBot/` (`.ino`,
  `InverseKinematics.cpp/.h`, `MotorControl.cpp/.h`, `RemoteStepControl.cpp/.h`); `CLAUDE.md`
  "Repository Structure" firmware-duplication note.

## Process / Meta

- [x] **Run the full thorough TODO-mining pass across all project docs** (flagged 2026-10-13,
  resolved 2026-10-13) — ran the full pass: mined `docs/PROJECT_LOGBOOK.md` in full (all ~1750
  lines, back to 19/08/2026), `docs/EVALUATION_STRATEGY.md`, every file in
  `host_software/ml_jetson_vla/docs/`, `fpga/docs/optical_bridge_interface_spec_draft.md`, and
  `CLAUDE.md`'s own tables. Every candidate item was checked against current code/docs before
  being added (e.g. confirmed `vision_calibration.py` still unreferenced in
  `run_jetson_standalone.py`, confirmed `maybe_auto_hold()` still reads raw `marker_coords`,
  confirmed no jerk/disturbance/drop-detection code exists in `evaluate_system_control.py`,
  confirmed the 2026-10-09 `run_live()` label/instruction bug was already fixed same day so it
  was NOT re-added). Added 11 new genuinely-still-open items above (7 Vision/Multimodal, 2
  evaluation-pipeline gaps filed under Multimodal, 1 Hardware/firmware, 1 Multimodal
  terminology gap). Checked roughly 15-20 historical "flagged"/"not yet done" items from the
  logbook (08/2026-10/2026) and did NOT add them because later entries in the logbook itself, or
  a direct code check, showed them already resolved — notably: the red/blue hue-wraparound bug
  (fixed 2026-10-13, `_circular_mean_hue()`), the black/grey bin ordering bug in
  `marker_tracker.py` (fixed 2026-10-07, and that module is not on the live inference path
  anyway), the `run_experiment.py` `run_live()` marker-label-vs-instruction mismatch (fixed
  2026-10-09), the stm32 telemetry-proposal-not-merged gap in `ARCHITECTURE.md` (superseded —
  the new `AngleStepControl.cpp`/`TouchProbe.cpp` uplink is confirmed working live, H1/H2
  passed 2026-10-13), the lab-partner Qwen model "not in our possession" blocker (superseded —
  the project now runs its own downloaded Qwen2.5-VL-3B-Instruct checkpoint), and the entire
  2026-09-15 multi-head-Qwen (`MULTI_HEAD_ARCHITECTURE_SPEC.md`) design path (superseded by the
  October hybrid ACT/SmolVLA fast-layer architecture actually built and tested — no code in this
  repo implements the custom Head B/C design beyond a self-test skeleton).
