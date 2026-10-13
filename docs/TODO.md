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

- [x] **Marker colour classification needs a robust design, not another hardcoded fix**
  (flagged 2026-10-13, robust fix shipped same day, owner: ml-vision) — rather than a third
  bin-tuning pass, `MarkerClassifier` (`host_software/ml_vision/core/marker_classifier.py`) gained
  an opt-in manifest-position identification tier: a static marker's colour is already known by
  design for whatever sheet is physically mounted (`hardware/platform_templates/*_manifest.json`),
  and the classifier already computes each blob's ArUco-homography-derived (x_mm, y_mm) — matching
  that position against the active sheet's known feature slots (15mm radius, largest-blob-first
  claim) and reading `color` straight off the manifest is lighting/ink/camera invariant by
  construction, since it never inspects a pixel. Falls back to the original HSV bins when no
  manifest is supplied or no slot matches — `manifest_path=None` by default, so every existing
  call site is byte-for-byte unaffected unless it opts in. **Real measured numbers, same validated
  1200-frame harness** (`marker_classifier_green_fallback_sweep.py`, `session_20260810_114330`):
  shipped-defaults baseline reproduced exactly (black 1124/1200 93.7%, blue_triangle 595/1181
  50.4%, yellow_square 1154/1198 96.3%, green_hexagon 1145/1191 96.1%, red_triangle 1169/1196
  97.7% — byte-identical to the pre-change numbers, confirming zero regression to the default
  path); with `manifest_path` supplied: black 1196/1200 99.7%, blue_triangle 1181/1181 **100%**,
  yellow_square 1196/1198 99.8%, green_hexagon 1189/1191 99.8%, red_triangle 1196/1196 **100%** —
  improvement on every color, no regression on any. New tests:
  `host_software/ml_vision/tests/test_marker_classifier_manifest.py` (6 tests, including two that
  replay the exact real HSV values logged for `aruco_markers_01`'s misclassified black/green
  blobs and confirm the manifest tier resolves them correctly). Full `ml_vision/tests/` suite and
  `ml_jetson_vla/tests/test_small_cnn_grounder_cpu.py` (a real consumer) re-run clean, 0
  regressions. **Not yet done**: wiring `manifest_path=` into the actual live call sites
  (`run_jetson_standalone.py`, `main_onnx_shared_vision_*_audio.py`, `small_cnn_grounder.py`) and
  live-Jetson hardware re-verification on `aruco_markers_01` — out of this fix's file-boundary
  scope (`host_software/ml_vision/` only); see new item below. Source: `docs/PROJECT_LOGBOOK.md`
  13/10/2026 entries (`f88877e`, `9c76748`); fix logged 13/10/2026, "Marker Classifier:
  Manifest-Position Identification Tier Shipped".
- [ ] **Marker-classifier manifest-position fix: now wired into all 4 live call sites; live-Jetson
  confirmation still outstanding** `[JETSON-HW]` (flagged 2026-10-13, updated same day) —
  `small_cnn_grounder.py`, `run_jetson_standalone.py`, `main_onnx_shared_vision_audio.py`, and
  `main_onnx_shared_vision_nemo_audio.py` all now pass `manifest_path=`/`--marker-manifest`
  (defaulting to `aruco_markers_01_manifest.json`, confirmed physically mounted today) into
  `MarkerClassifier`. The dispatched `ml-vision` agent correctly refused to cross its own write
  boundary (`host_software/ml_vision/` only — none of the 3 remaining call sites actually live
  there, a mistake in how the task was scoped, not the agent's error) and drafted exact patches
  instead of silently skipping or overstepping; the orchestrator applied them directly. Verified:
  all 3 edited files `py_compile` clean; full `ml_jetson_vla/tests/` suite re-run (154 passed, 1
  pre-existing skip, 0 regressions); `ml_vision/tests/` re-run (same pre-existing baseline, 2
  failed/11 passed/6 errors, all unrelated); a real `JetsonExpertPolicy` constructed directly
  (real ONNX session, `dummy_audio=True`) confirmed to load the correct 5 real `aruco_markers_01`
  features, not mocked. **Still genuinely outstanding**: live-Jetson confirmation
  (`diagnose_small_cnn_grounder_live.py --instruction go_black`-style re-run against the real
  mounted sheet) has not happened for ANY of the 4 wired call sites yet, including the one wired
  yesterday — nothing here should be read as hardware-confirmed. Source: `docs/PROJECT_LOGBOOK.md`
  13/10/2026, "Manifest-position colour-ID fix wired into all 4 live call sites."

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
