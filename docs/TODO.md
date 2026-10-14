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
  **2026-10-14 real fix built (pixel-only, no ground truth)**: new
  `core/color_calibration.py`'s `AdaptiveColorCalibrator` — a per-session, pixel-only calibrator
  that tracks each spatially-stable blob slot across frames and assigns colour via *relative*
  reasoning (darkest slot -> black by brightness rank; remaining slots -> whichever contiguous
  run of the fixed hue order `[red,yellow,green,blue]` best fits the *whole* observed hue set
  together, not nearest-centre-per-point) — never reads the manifest. Wired into
  `MarkerClassifier` as a new opt-in `adaptive_calibrator=` tier (between the still-eval-only
  `manifest_path` tier and the original HSV-bin fallback); bare `MarkerClassifier()` construction
  verified byte-for-byte unchanged (every existing call site still constructs it bare).
  Independently re-verified (not just trusting the agent's report): full `ml_vision/tests/` run
  -> **42 passed, 2 skipped, 2 errors** (same 2 pre-existing unrelated errors as before — see
  this file's own note on `test_preprocessor.py`/`test_yolo_tracking.py`), 17 new tests all
  passing. Real numbers on `session_20260810_114330` (1200 real `_03` frames): adaptive
  calibration matches or beats static bins on every colour (e.g. `blue_triangle` 3.9%->99.8%,
  `black_circle` 95.2%->100.0%). On the actual real documented `_01` failure (the one logged
  frame from the 13/10 entry): 3/5 correct (static bins) -> **5/5 correct** (adaptive
  calibration), with zero manifest access. **Known, documented, not-yet-solved limitation**:
  assumes each position slot is a visually distinct colour — true for `aruco_markers_01`/`_03`
  (confirmed, real data), **wrong for `aruco_markers_02`** (4 real shapes are all "blue" by
  design) — no real `_02` capture exists yet to design against.
  **2026-10-14, wired into all 4 live call sites (ml-multimodal, reviewed not rubber-stamped)**:
  `small_cnn_grounder.py`'s `SmallCnnGroundingPolicy` and `run_jetson_standalone.py`'s
  `JetsonExpertPolicy` each now construct `self.adaptive_calibrator = AdaptiveColorCalibrator()`
  in `__init__` and pass it to their own `MarkerClassifier(...)`; `main_onnx_shared_vision_audio.py`
  and `main_onnx_shared_vision_nemo_audio.py` (root-level, touched by ml-multimodal for wiring
  consistency with the other 2) do the same at their single `marker_classifier = MarkerClassifier(...)`
  construction site in `main()`. One calibrator instance per `MarkerClassifier` instance in every
  case — never shared/global. Compatibility checked, not assumed: all 4 sites construct their
  `MarkerClassifier` exactly ONCE per process and drive it from a single continuous loop for that
  process's whole lifetime (`run_jetson_standalone.py`/both `main_onnx_shared_vision_*.py`'s
  `while True:`; `SmallCnnGroundingPolicy`'s instance persists for `HybridSmallCnnActPolicy`'s
  whole life, surviving `_orchestrator.reset()` calls between episodes within the same process,
  which is correct since the physical sheet doesn't change between episodes) — exactly the
  "one sheet mounted per session" shape the calibrator's warm-up design assumes. No call site
  wires `calibrator.reset()` into its own `.reset()`/episode-reset path, deliberately: a real
  sheet swap requires restarting the process (which naturally creates a fresh calibrator), and
  resetting on every episode would discard an already-good calibration for no reason. The live
  sheet these 4 sites actually target (`aruco_markers_01`, per the sheet-mismatch this item is
  about) has 4 distinct chromatic colours + black — the calibrator's supported case, not the
  known-wrong `aruco_markers_02` (4-same-colour) case, so that documented limitation does not
  block this wiring. Verified for real: `pytest host_software/ml_vision/tests/` ->
  **42 passed, 2 skipped, 2 errors** (same 2 pre-existing `test_preprocessor.py`/
  `test_yolo_tracking.py` errors, unchanged, confirming the untouched `ml_vision/` files still
  behave identically); `pytest host_software/ml_jetson_vla/tests/` -> **155 passed, 1 skipped**
  (150 prior + 5 new: 2 added to `test_small_cnn_grounder_cpu.py`, 3 in a new
  `test_run_jetson_standalone_cpu.py` that does a REAL, unmocked `JetsonExpertPolicy` construction
  — real ONNX session, real `MarkerClassifier`, real `AdaptiveColorCalibrator` — confirming
  per-instance wiring and that `.reset()` doesn't discard the calibrator). The 2 root-level
  `main_onnx_shared_vision_*.py` files have no pre-existing test harness at all (hardware-driving
  `main()` scripts, never covered by any test file) — their wiring was verified by `py_compile`
  plus direct code review using the exact same `MarkerClassifier(..., adaptive_calibrator=...)`
  call shape already exercised by the suites above, not by a new dedicated test file (building
  one was judged out of scope for a wiring-only change).
  **Still genuinely open**: (1) live-Jetson re-verification against genuinely independent real
  `_01` frames (the numbers above still replay the one logged frame to satisfy warm-up, not
  independent captures) — needs real hardware access. (2) `aruco_markers_02`'s 4-same-colour
  case remains unhandled if that sheet is ever physically mounted at one of these 4 call sites —
  not currently the case, so not a blocker today, but flagged for whoever changes the mounted
  sheet next.
  Source: `host_software/ml_vision/core/color_calibration.py`,
  `host_software/ml_vision/core/marker_classifier.py` (`adaptive_calibrator` tier, additive);
  `host_software/ml_jetson_vla/core/small_cnn_grounder.py`,
  `host_software/ml_jetson_vla/runtime/run_jetson_standalone.py`,
  `host_software/main_onnx_shared_vision_audio.py`,
  `host_software/main_onnx_shared_vision_nemo_audio.py` (wiring, additive);
  `host_software/ml_jetson_vla/tests/test_small_cnn_grounder_cpu.py`,
  `host_software/ml_jetson_vla/tests/test_run_jetson_standalone_cpu.py` (new wiring tests).
- [x] **The "permanently forbidden" `fliplr` augmentation bug was still live in 2 YOLO training
  scripts + 1 Colab notebook, with real checkpoints trained while it was active** (flagged
  2026-10-14, fixed same day, commit `db5fee2`) — `train_yolov8_platform_pose_markers.py`,
  `train_yolov8_platform_corners.py`, and `colab_train_yolo.ipynb` cell 3 now all have
  `fliplr=0.0` with the same explanatory comment as the original `train_yolo.py` fix. Verified:
  `grep fliplr` across `training/*.py` + the notebook shows no remaining `0.5`; all 3 files
  `py_compile`/JSON-parse clean. **Still open, `[HUMAN]` call**: `yolov8_platform_pose_markers_
  0728_v3`/`yolov8_platform_corners_0730_v1` (the real checkpoints these scripts' `name=` points
  at) were trained with the corrupting augmentation active and need retraining before being
  trusted for any Arm 1 comparison — not retrained here, a real compute/time cost. Source:
  `docs/PROJECT_LOGBOOK.md` 14/10/2026.
- [x] **2 tests for the project's most historically bug-prone code paths were stale, testing an
  API/schema that no longer exists** (flagged 2026-10-14, fixed same day, commit `150f805`) —
  `test_auto_label_shared_vision.py` now calls the real `load_manifest_full()` with a schema
  that actually matches, plus a new `ValueError`-on-bad-schema regression test, a
  `project_points()` test using a non-trivial homography (the old identity-matrix version would
  have passed even if the function ignored its argument), and a parametrized per-shape
  `render_marker_mask()` test checking real polygon-area ratios — catching the previously-shipped
  "silent circle fallback for non-triangle shapes" bug class, not just `mask.sum() > 0`.
  `test_shared_vision_training.py` now asserts the real 4-tuple dataset item and 3-tuple model
  output, including heatmap semantics (non-degenerate on a lit mask, all-zero on an empty one —
  the exact "degenerate target" bug class from 2026-08-12). Source: `docs/PROJECT_LOGBOOK.md`
  14/10/2026.
- [x] **A genuinely new, previously-unknown bug found while fixing the above: `ultralytics`
  globally monkeypatches `cv2.imread` at import time, silently corrupting grayscale mask reads
  process-wide** (flagged + fixed 2026-10-14, commit `e41a463`) — confirmed directly with a
  6-line repro: `cv2.imread(path, cv2.IMREAD_GRAYSCALE)` returns `(64,64)` before
  `from ultralytics import YOLO` runs anywhere in the process, and `(64,64,1)` after, silently,
  for the rest of that process's life (`ultralytics/utils/__init__.py`'s global
  `cv2.imread, cv2.imwrite, cv2.imshow = imread, imwrite, imshow`, and its replacement `imread()`
  unconditionally appends a trailing channel axis). Every shared-vision mask is square
  (128x128), so the extra axis does NOT raise downstream — it silently mis-broadcasts against
  `mask_logits` in loss/IoU computation instead, a worse failure mode than a crash. No file
  currently imports both `ultralytics` and a shared-vision mask reader in the same process, but
  the small/medium model classes are actively developed side by side for the still-open Arm 1
  comparison — a live landmine, not hypothetical. **Fixed**: squeezed any trailing singleton
  channel axis back to 2D immediately after every `cv2.imread(..., cv2.IMREAD_GRAYSCALE)` call
  in `training/shared_vision_dataset.py`, `data_processing/merge_shared_vision_sessions.py` (the
  most load-bearing — `np.maximum()`-combining a corrupted `(h,w,1)` mask against `(h,w)` would
  silently explode to `(h,w,w)` rather than erroring, since these masks are square),
  `data_processing/verify_shared_vision_labels.py`, `experiments/trial_augmentation_strategies.py`.
  No-op when `ultralytics` isn't imported. New regression test monkeypatches `cv2.imread` the
  same way `ultralytics` does (without the slow real import) so this is pinned down, not left to
  accident of test order. **Watch for**: any future file adding a new
  `cv2.imread(..., cv2.IMREAD_GRAYSCALE)` call anywhere `ultralytics` might also get imported
  in-process should carry the same defensive squeeze. Source: `docs/PROJECT_LOGBOOK.md`
  14/10/2026.
- [x] **Running `pytest host_software/ml_vision/tests/` with no flags hard-aborted with zero
  tests run — a false-confidence trap for "I ran the tests, they're clean"** (flagged 2026-10-14,
  fixed same day, commit `fb92add`) — `test_combined_controller.py` deleted (confirmed via
  filesystem search + `git log --all` that the module it imported,
  `combined_audio_video_controller`, never existed anywhere in the repo or its history — nothing
  to restore). `test_openvino_yolo.py`/`test_yolo_ov.py` confirmed the referenced
  `yolov8_platform_markers_v{1,2}` export has never existed either (current listing, archive, or
  any commit); rather than delete, wrapped each in a real `test_*` function behind
  `pytest.mark.skipif` on the artifact's existence, paths resolved relative to the repo root —
  they skip cleanly now and will actually run if that export ever reappears. **Real pass counts**
  (same machine, same flags): before, `pytest host_software/ml_vision/tests/` (no flags) ->
  `Interrupted: 3 errors during collection`, 0 tests ran; with
  `--continue-on-collection-errors` -> `2 failed, 12 passed, 5 errors`. After, flagless ->
  `23 passed, 2 skipped, 2 errors`, no abort. The 2 remaining errors
  (`test_preprocessor.py::test_preprocess`, `test_yolo_tracking.py::test_yolo`) are a separate,
  pre-existing, out-of-scope issue (bare scripts mistaken for pytest tests, same class as the
  2 OpenVINO files but failing on missing fixtures rather than collection-time import errors) --
  not touched, flagged as a worthwhile follow-up if this gets picked up again. Source:
  `docs/PROJECT_LOGBOOK.md` 14/10/2026.

## Audio (`ml-audio`)

- [x] **Unbounded live audio chunk queue with blocking inference — no backpressure or staleness
  detection** (flagged 2026-10-14, fixed same day) — `nemo_live_receiver.py`'s
  `self.chunk_queue = queue.Queue()` had no `maxsize`; `audio_receiver_pytorch.py` had the
  identical shape. A single consumer thread runs a blocking model forward pass before pulling
  the next chunk; if that ever took longer on average than `step_seconds` (0.2s default) —
  plausible on CPU, especially sharing the Jetson with vision inference — chunks would silently
  accumulate with nothing bounding the queue or logging it, letting the command stream drift
  seconds behind real time while still looking "live" — exactly the class of bug this project's
  own non-blocking-execution constraint (`CLAUDE.md`) exists to prevent. **Fixed in both files**:
  `chunk_queue` now bounded at `CHUNK_QUEUE_MAXSIZE=5` (1s of backlog at the default
  `step_seconds`); a new `_put_chunk_dropping_oldest()` helper drops the OLDEST buffered chunk
  on overflow rather than blocking the real-time audio callback, with every 10th drop logged
  (rate-limited, not every one, to avoid log-flooding during a sustained stall) so a real
  backlog is visible instead of silent. Deliberately left the file-reader loop's own
  `chunk_queue.put()` (not `put_nowait`) unchanged in both files -- that's a self-throttling,
  reproducible offline-replay path, not a real-time callback that must never block; bounding the
  queue just means it now self-paces slightly earlier under a sustained stall, which is correct
  behavior there, not a regression. 6 new tests
  (`ml_audio/tests/test_chunk_queue_bounding.py`), all passing — confirming the bound, the
  drop-oldest-not-newest ordering, and the rate-limited warning. Source:
  `host_software/ml_audio/evaluations/nemo_live_receiver.py`;
  `host_software/ml_audio/audio_receiver_pytorch.py`.
- [x] **Module-level mutable `LABEL_NAMES` global in the legacy PyTorch receiver — same bug
  class that already cost ~80 accuracy points in production** (flagged 2026-10-14, fixed same
  day) — `audio_receiver_pytorch.py`'s `AudioCommandReceiver.__init__` no longer does
  `global LABEL_NAMES`; all 3 assignment sites and the one read site now use `self.labels`
  (the NeMo receiver already did this correctly). 3 new tests
  (`ml_audio/tests/test_audio_receiver_pytorch_labels.py`): two receivers built against
  different checkpoints in the same process no longer clobber each other's labels, the module
  global is never mutated, and the no-`model_path` fallback still works. Had to mock both
  `AudioCommandClassifier` (bakes in fixed registered-buffer weights, not freely constructible
  with arbitrary shapes) and `sd.InputStream` — the first unmocked run of this test genuinely
  opened this machine's real microphone, confirming the constructor had no test-safe path.
  Source: `host_software/ml_audio/audio_receiver_pytorch.py`.
- [ ] **Production entry point's default audio checkpoint is still the preliminary, never
  live-stream-scored one — contradicts `CLAUDE.md`'s own "Current State" narrative** (flagged
  2026-10-14, owner: ml-audio, NOT fixed -- see below) — `main_onnx_shared_vision_nemo_audio.py`'s
  `DEFAULT_NEMO_MODEL` points at `nemo_matchboxnet_3x1x64_noisemix_seed0` (the code's own
  comment already calls this "Preliminary/half-trained"), while `CLAUDE.md` describes
  `nemo_matchboxnet_v1` (8-9/10 live-stream, two independent runs) as the established result.
  **Checked the module docstring before touching this (2026-10-14): this was wired up at the
  user's own explicit request for integration-testing on 2026-09-15/16, not a stale oversight.**
  Deliberately NOT flipped — reversing an explicit prior instruction just because a month has
  passed is a confirm-with-user call, not a trivial fix an agent should make unilaterally. `[HUMAN]`
  to confirm either way; if confirmed stale, fix is a one-line default change, ideally after
  running `evaluate_nemo_live_receiver_stream.py` against the seed0 checkpoint first to know
  whether it's worth keeping as a reference point at all. Source:
  `host_software/main_onnx_shared_vision_nemo_audio.py` lines 97-102, 128.
- [x] **No automated regression coverage of the actual production audio inference/preprocessing
  path** (flagged 2026-10-14, owner: ml-audio) — `ml_audio/tests/test_audio.py` opens a mic
  stream for 2 seconds and prints whether it errored; zero assertions, not wired to any
  classifier/DSP code. `test_nemo_noise_mix.py` is a real smoke test but only exercises
  training-time noise-perturbation wiring (needs a separate conda env), not a regression guard
  on `_process_loop`/`audio_dsp.py`. This project has already paid for two severe
  train/production-preprocessing-divergence bugs (wrong label order; mismatched crop/
  renorm/noise-profile), both only caught by manual A/B investigation -- nothing today would
  fail if either regressed again. **Fixed 2026-10-14**: new
  `host_software/ml_audio/tests/test_production_pipeline_parity.py` (282 lines, 4 tests) —
  `OfflineLiveFeatureParityTests` drives a real `AudioCommandReceiver` (model/mic mocked) through
  its real `chunk_queue` -> `_process_loop` path over all 11 real clips in
  `ml_audio/data/01_evaluation_samples/*.wav`, asserting the spectrogram actually fed to the
  model is numerically identical (`torch.allclose`, atol=1e-6) to the offline
  `evaluate_audio_classifier.py` path's output for the same clip; `LabelOrderSanityTests` guards
  the exact historical ~7.4%-accuracy label-order regression, including the hardcoded 12-class
  fallback branch. Verified the tests have real teeth (not vacuous): temporarily reintroduced
  both historical bug classes (a renorm scale factor; a non-alphabetical fallback order) and
  confirmed both failed loudly with clear diagnostics, then reverted (confirmed clean via
  `git status`). Independently re-run (not just trusting the agent's report):
  `python -m unittest discover -s host_software/ml_audio/tests -p "test_*.py"` -> **13 tests,
  OK** (4 new + 9 pre-existing). **Scope note, explicitly flagged by ml-audio and carried
  forward here**: this covers the custom Conv1D CNN path only
  (`audio_receiver_pytorch.py`/`evaluate_audio_classifier.py`) per the TODO item's original
  wording — the pretrained NeMo MatchboxNet path (the actually-deployed production model per
  `CLAUDE.md`'s Current State table) still has no equivalent coverage; see the new item below.
  Source: `host_software/ml_audio/tests/test_audio.py`; `audio_eval_notebook_refactor_plan.md`
  "Evaluation Extraction" section.
- [ ] **NeMo MatchboxNet production path has no parity/label-order regression coverage
  equivalent to the Conv1D path's new tests** (flagged 2026-10-14, owner: ml-audio) — the item
  above only covers the custom Conv1D CNN path; the NeMo path
  (`evaluations/nemo_live_receiver.py`, `main_onnx_shared_vision_nemo_audio.py`) is the model
  CLAUDE.md's Current State table names as actually in production, and is unguarded. Needs its
  own fixture/assertion shape, not a drop-in extension of `test_production_pipeline_parity.py`:
  it requires the separate NeMo conda env `test_nemo_noise_mix.py` already needs, and its label
  order comes from `model.cfg.labels` inside a restored `.nemo` checkpoint rather than a sibling
  `labels.json`.

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
- [~] **Task Success Rate can't distinguish a dropped ball from a trial that never settled**
  `[design + wiring done 2026-10-14; JETSON-HW final confirmation still open]`
  (flagged 2026-09-18, owner: ml-multimodal/orchestration) — **the design-and-offline-validation
  half of this is now done, and wiring into `evaluate_system_control.py` is also done
  (orchestration, since the file sits outside ml-multimodal's write boundary).** Chosen signal:
  `touch_valid` (`TouchProbe.cpp`'s pressure-vote flag,
  deliberately logged even on loss of contact). The other two candidates don't work: vision-side
  ball tracking's "lost the ball" frames are never logged at all (by design — `act()` skips
  `no_ball` frames entirely), and `touch_x`/`touch_y` carries no bounds/sentinel signal on drop
  (firmware holds the last real position, only `touch_valid` flips). Implemented as a standalone
  module (same reason as the jerk metric below — `host_software/evaluations/` is read-only for
  this agent): `host_software/ml_multimodal/evaluations/ball_drop_detection.py`
  (`ball_drop_metrics_for_segments()`), 14 passing tests in
  `ml_multimodal/tests/test_ball_drop_detection.py`. Flags a trial segment as a drop when contact,
  established earlier in that segment, is lost and never recovers for ≥1000ms before the segment
  ends — threshold picked from a real, order-of-magnitude gap (≤120ms ordinary boundary noise vs.
  ≥2760ms smallest real candidate) found in the one real audit possible with existing data.
  **Real-data result** (all 15 `host_software/data/01_bronze/evaluation/ground_truth_jetson_*.csv`
  sessions — the only sessions in this repo whose schema carries `touch_valid`): 3 of ~210 trial
  segments flagged, all from anomalously short-duration sessions consistent with (not proof of) an
  early-stopped run. **Honest limit, stated plainly per this task's own instruction: no session in
  this repo carries a human-confirmed "a drop happened here" label**, so this is the strongest
  offline proxy available, not a confirmed count — real-drop confirmation is genuinely
  JETSON-HW-gated, same as this item's own original tag said. A second, independent finding from
  the same audit: 2 of those 3 candidates are **currently counted as successes** by
  `find_settling_time_ms()` today (it never checks `touch_valid`), which the module's
  `Settled_Trials_Overridden_By_Drop`/`Task_Success_Rate_Percent_Drop_Corrected` fields correct,
  not just annotate. **Wired into `evaluate_system_control.py` 2026-10-14 (orchestration pass,
  following the module's own "Integration notes")**: `compute_metrics()` now tracks a
  `settled_mask` list (`None`/`True`/`False` per segment, aligned with `segments`) through its
  existing per-segment loop and merges `**ball_drop_metrics_for_segments(times, touch_valid_col,
  segments, settled_mask)` into its return dict (after `jerk_metrics_for_segments`, so its more
  complete `Task_Success_Rate_Caveat` wins the dict-literal duplicate-key override).
  `Task_Success_Rate_Percent` itself is deliberately left unchanged for historical comparability
  — `Task_Success_Rate_Percent_Drop_Corrected` is the corrected figure. **Verified end-to-end
  against real data** (not just unit tests): ran `compute_metrics()` on all 15 real
  `ground_truth_jetson_*.csv` sessions — flags the same 3 segments the standalone module's own
  audit found, with `Settled_Trials_Overridden_By_Drop` matching exactly (2 of 3). Full
  `ml_multimodal/tests/` suite: **29 passed** (5 dataset + 10 jerk/smoothness + 14 new ball-drop,
  0 regressions). **Still open**: (1) `touch_valid` is missing from two of this project's
  telemetry schemas — `ml_jetson_vla/runtime/session_recorder.py`'s Track 4
  `TELEMETRY_CSV_FIELDS` parses it already (`SessionTouchTap._handle_line()`) but never writes it
  to `telemetry.csv`, and the expert-pipeline PID schema
  (`data/04_evaluation_expert/expert_evaluation_run*.csv`) has no equivalent column at all — so
  this signal isn't yet available for every arm; (2) the JETSON-HW confirmation itself (no
  session in this repo carries a human-confirmed "a drop happened here" label — the 3 flagged
  segments are the strongest available offline proxy, not a confirmed count).
  Source: `docs/EVALUATION_STRATEGY.md` "Evaluation Pipeline" + "Extended Metrics" sections (both
  updated in place 2026-10-14); `host_software/ml_multimodal/evaluations/ball_drop_detection.py`'s
  own module docstring has the full audit; `host_software/evaluations/evaluate_system_control.py`
  (wiring).
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
- [x] **`VLADataset`'s vocabulary never matched real audio-command strings — language
  conditioning was a no-op for every real sample** (flagged 2026-10-14, fixed same day) —
  `core/dataset.py`'s `self.vocab` used space-separated keys (`"go red"`, `"go blue"`, ...); the
  real recognized vocabulary (confirmed directly against 10 real `session_jetson_track4_*`
  telemetry CSVs) is underscore-separated and matches `minimal_vlm_policy.py`'s
  `COLOR_COMMANDS`/`DIRECTIONAL_COMMANDS` exactly. Fixed: `VOCAB` now has all 10 real strings
  (`hold`, `go_red`, `go_green`, `go_yellow`, `go_black`, `forward`, `backward`, `left`, `right`,
  `stop`) at distinct indices. 5 new tests (`ml_multimodal/tests/test_dataset.py`), all passing
  — confirming every real command resolves to its own index (not silently aliasing "hold"), and
  the old dead space-separated keys are gone. **Any existing `models/vla_v1/best_vla.pth`
  checkpoint was trained under the old, broken mapping and needs retraining** to actually
  benefit from this fix — not done here (a real compute cost, not an agent's call to schedule).
  Source: `host_software/ml_multimodal/core/dataset.py`.
- [ ] **`run_eval_baseline_vla.py`/`run_eval_our_vla.py` are 100% mocked — cheap mitigation
  applied 2026-10-14, real fix still substantial and NOT done** (flagged 2026-10-14) — both
  scripts never touch a camera or the firmware: mock marker coordinates, mock touch state
  (`tx+8.0, ty+2.0`), a zero image tensor; `run_eval_our_vla.py` comments out the one line that
  would load different weights than the baseline, so both literally run the same untrained,
  randomly-initialized model. **Mitigation applied**: output files renamed to
  `labels_sequential_{baseline,our}_vla_MOCK.csv` (an `_MOCK` suffix specifically so they can't
  be mistaken for real telemetry if fed into `evaluate_system_control.py`), a loud runtime
  warning added to `VLAEvaluator.__init__`, and `train_vla.py`'s `fine_tune_rl()` now raises
  `NotImplementedError` with a clear explanation instead of printing "VLA Model is now fully
  prepared for deployment and RL refinement!" when it never actually ran PPO/REINFORCE or saved
  a checkpoint. `evaluate_system_control.py`'s own `--help` example path updated to match.
  Verified: all 3 edited files `py_compile` clean; confirmed `fine_tune_rl` has exactly one real
  caller (its own `__main__`), so this is a safe, isolated change. **Still substantial, not
  attempted**: the real fix needs actual camera/serial integration and a genuine Stage-2 RL
  implementation before "our VLA" means anything different from "baseline." Source:
  `host_software/ml_multimodal/evaluations/run_eval_baseline_vla.py` lines 17-18, 65-75,
  100-112; `run_eval_our_vla.py` lines 12-13; `train_vla.py` lines 112-129.
- [x] **`VLADataset.__getitem__`'s bare `except:` silently substituted a blank image with zero
  logging** (flagged 2026-10-14, fixed same day) — narrowed to `(FileNotFoundError, OSError)`
  (no longer catches `KeyboardInterrupt`/`SystemExit`), now counts and logs every fallback via
  `self._image_fallback_count`. Verified by a new test confirming the counter increments
  correctly across repeated failures (`ml_multimodal/tests/test_dataset.py`). Source:
  `host_software/ml_multimodal/core/dataset.py`.
- [~] **Jetson expert-pipeline control-net port (`core/control_net.py`) is further along than
  `CLAUDE.md`'s current framing suggests, and its velocity-filter timing is unverified against
  real achieved loop rate** `[the dt-comparison analysis itself needs no hardware -- existing
  telemetry has what's needed; completing the comparison + trusting it on real motors is
  JETSON-HW]` (flagged 2026-10-14, owner: ml-multimodal) — this module is a real, working port already wired into
  `runtime/run_jetson_standalone.py --remote-control` (Phase B), with matching firmware already
  built and functioning at `firmware/stm32_jetson_remote_control/BallBalancingBot/
  RemoteStepControl.cpp` (confirmed: parses `S,<a>,<b>,<c>`, clamps to +/-98 steps, fails safe to
  level after 3s stale). Worth noting since the project's own current-state framing describes
  this track as earlier-stage than the code actually is. Real open risk: `ControlNet.infer()`'s
  `VEL_FILTER_ALPHA=0.35` EMA (mirroring `RLControl.cpp`) is applied identically regardless of
  `actual_dt` -- a fixed-alpha EMA's effective time constant is dt-dependent, so if the real
  Jetson deployment loop's dt distribution differs from the training data's, the filtered
  velocity could lag more than the policy was ever exposed to.
  **Correction to this item's own original framing (2026-10-14):** `session_jetson_track4_*/
  telemetry.csv` is NOT an STM32-native/hardware-rate signal as first assumed -- checked
  `data_collection/collect_vla_data.py` directly: it's a laptop-webcam collection loop
  (`cv2.CAP_PROP_FPS=30`, `cap.read()`-bound), i.e. the same *category* of camera-bound host
  loop a real Jetson deployment run is, not the STM32's internal fast loop. The real open
  question is "does the Jetson's camera-bound loop run at a different dt than the
  training-data collection loop did," not "hardware loop vs. software loop."
  **Done so far (no hardware needed):** built `ml_jetson_vla/experiments/analyze_control_loop_dt.py`
  (reusable dt-histogram tool, same `host_timestamp_ms` column on both sides) and ran it against
  real training-data telemetry (`session_jetson_track4_20260915_151627/telemetry.csv`, 2596
  ticks): mean dt **41.37ms (std 2.91ms, implied ~24.2Hz)**, p50=40ms, p90=45ms, p99=50ms,
  max=57ms -- i.e. the training data's own host loop never ran anywhere close to the 30fps cap,
  let alone 120Hz.
  **Still blocked, JETSON-HW:** no real `run_policy_live.py`/`run_jetson_standalone.py`
  `.telemetry.csv` from an actual Jetson deployment run exists on this machine yet (checked
  `ml_jetson_vla/reports/` -- the earlier test session's JSON summaries were pasted into chat,
  not the sidecar CSVs) -- copy one back from the Jetson and re-run the tool with
  `--deployment <path>` to get the real comparison number. Source:
  `host_software/ml_jetson_vla/core/control_net.py` (`VEL_FILTER_ALPHA`, line 49);
  `host_software/ml_jetson_vla/experiments/analyze_control_loop_dt.py` (new);
  `firmware/stm32_jetson_remote_control/BallBalancingBot/RemoteStepControl.cpp`.

## FPGA

- [ ] **Arm 3 framing decision** `[HUMAN]` (flagged 2026-10-13) — is Arm 3 a real fabricated photonic
  chip, a simulation/co-design-only arm, or a commercial product? Everything in the first-draft
  interface spec is written conditional on "real chip" since that's the only scenario with
  concrete numbers available -- this is a one-line question that unblocks or reframes the whole
  spec, not yet answered. Source: `fpga/docs/optical_bridge_interface_spec_draft.md` section 1.
  **2026-10-14 update**: still unanswered, still blocking any real RTL/register-map design — but
  a new `fpga/optical_bridge/` directory now holds a clearly-flagged PRELIMINARY/SPECULATIVE
  behavioral Verilog sketch (AXI4-Lite register wrapper + calibration-table store + write-DAC/
  settle/read-ADC FSM, parameterized N/C/DAC_WIDTH/ADC_WIDTH/settle-cycles with every value
  called out as a placeholder) built explicitly as scaffolding for whenever this decision lands,
  not a step toward deciding it. Testbench run for real via Vivado xsim 2025.2 (`xvlog`/`xelab`/
  `xsim`, not `iverilog` — not on this machine's PATH despite `fpga/testbenches/camera_i2c/
  run_tests.sh` assuming it, a separate pre-existing inconsistency, not fixed here): **all
  checks passed, 0 errors** (verified directly against the real `xsim.log`, not just the
  agent's report). See `fpga/optical_bridge/README.md` for the full placeholder-by-placeholder
  checklist of what must be real before this sketch becomes a real design.
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
- [x] **No reset path for `balance_controller`'s persistent integrator state, and `ik_solver`
  failed silently with no fault signal** `[RESOLVED 2026-10-14, owner: fpga — real tool-verified
  fix, not a guess]` — **ap_rst semantics, confirmed two ways**: (1) AMD's own UG1399
  ("Controlling Initialization and Reset Behavior") states the default `config_rtl -reset` level
  is `control` — only FSM/I-O-protocol registers reset on `ap_rst_n`; registers derived from
  static/global C variables are explicitly left un-reset by default. (2) Confirmed directly
  against this project's own real csynth run and its generated RTL (not just the doc): the
  *unmodified* source first failed to even compile under Vitis HLS 2025.2 (ambiguous `operator*`
  between `ap_fixed` and raw `double` literals, and an ambiguous `hls::atan2` call — see the DSP
  item below); after fixing those compile errors and re-running csynth with no reset directive,
  `balance_controller_csynth.rpt` logged `WARNING: [RTGEN 206-101] Register 'integ_x' is
  power-on initialization` (same for `error_prev_x`/`integ_y`/`error_prev_y`) — i.e. cleared only
  at FPGA configuration, not by `ap_rst_n` during operation, exactly as the doc predicted. **Fix
  applied and verified**: added `#pragma HLS RESET variable=error_prev_x/y` and
  `variable=integ_x/y` inside `balance_controller()` (`fpga/hls_hardware/PIDControllers.cpp`).
  Re-ran csynth and inspected the actual generated Verilog
  (`hls_proj/solution1/syn/verilog/balance_controller.v`, not just the report) — confirmed
  `integ_x`/`error_prev_x` now have `if (ap_rst_n_inv == 1'b1) integ_x <= 23'd0;` etc., i.e. a
  real synchronous clear tied to the block's existing `ap_rst_n`. **Not fully closed**: this only
  means asserting the whole block's reset now also clears the integrators — it does **not** give
  software a way to clear just the integrators on ball-drop/re-placement without resetting the
  whole control core. That's a separate, still-open design decision (a software-writable clear
  register vs. some other mechanism) this agent should not make unilaterally.
  **`ik_solver` fault signal — added**: `IKResult`/`CalculatedAngles` (`InverseKinematics.h`) now
  carry a `bool fault` field; `ik_solver`'s unreachable-target branch sets it true instead of
  silently returning `{0,0}`; `get_angles()` ORs all three legs' faults; `balance_controller()`
  now has a new `bool &out_fault` output (`s_axilite`, bundle `CTRL`) — confirmed it synthesizes
  cleanly in the same csynth run. Note: this is a new output port, so anything that later wires
  `balance_controller()` into `main.c`/a PS driver needs to account for the extra signature
  argument — nothing currently does (Domain 4 isn't wired up yet per `CLAUDE.md`), so no breakage
  today, but flagging for whoever does that wiring next.
  Source: `fpga/hls_hardware/PIDControllers.cpp`, `fpga/hls_hardware/InverseKinematics.{h,cpp}`,
  `fpga/hls_hardware/run_csynth.tcl` (new batch-build script used to verify all of this).
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
- [x] **Repeated, unshared transcendental HLS math calls against the 220-DSP ceiling** `[REAL
  csynth run completed 2026-10-14, owner: fpga — DSP is fine, but it surfaced a worse, real LUT
  problem the original item didn't anticipate]` — Ran actual `vitis_hls`/`vitis-run --mode hls`
  batch `csynth_design` (no standalone `vitis_hls` binary ships in this 2025.2 install; new script
  `fpga/hls_hardware/run_csynth.tcl` does it via `vitis-run --mode hls --tcl`, part string
  `xc7z020clg484-1` — the one this project's real Vivado builds actually use, NOT the
  `XC7Z020-1CSG484CES` string that appears in this agent's own role description, which doesn't
  match any real build log in this repo and should be treated as suspect).
  **As originally written, the file did not compile at all** — real `csynth` errors (HLS
  207-4090/207-4085), independent of resource sharing: ambiguous `operator*` between `ap_fixed`
  and bare `double` literals (`-d * 0.5` etc. in `calculate_triangle`/`get_angles`), and an
  ambiguous `hls::atan2` call because `hls::sqrt(r[0]*r[0]+r[1]*r[1])`'s HLS-promoted return type
  didn't exactly match `r[2]`'s `ap_fixed<32,16>`, so `hls_math.h`'s only exact-match
  `atan2(ap_fixed<W,I>,ap_fixed<W,I>)` template couldn't bind. Fixed by casting the literals to
  `(fixed_t)` and both `atan2` args to `(fixed_t)` explicitly (`InverseKinematics.cpp`). **Once it
  compiled**, `hls::sin/cos/sqrt/acos/atan2` on `ap_fixed<32,16>` synthesized fine — confirms that
  part of the item, but only after the above fix, not as originally written.
  **Real numbers, unshared** (as the TODO worried): `balance_controller_csynth.rpt` —
  136/220 DSP (62%, fits with margin) but **164443/53200 LUT (309% — ~3x over the real xc7z020's
  entire LUT fabric)**, driven by exactly the triplicated-`ik_solver` mechanism this item
  predicted (3 full separate instances, ~24 DSP/~52000 LUT each, confirmed in the report's
  Instance table). **So the 220-DSP ceiling itself was never the real risk — LUT exhaustion from
  the same root cause is**, and is far more severe than "unverified."
  **Fix applied and verified**: `#pragma HLS ALLOCATION function instances=ik_solver limit=1` in
  `get_angles()` (`InverseKinematics.cpp`). First attempt used the wrong argument order
  (`instances=... limit=... function`) and was rejected by a real csynth run (`WARNING: [HLS
  207-7036] unexpected pragma argument 'ik_solver', expects function/operation`) — corrected to
  `ALLOCATION function instances=... limit=...` per UG1399's documented pragma template, then
  re-verified: the report now shows a single shared `ik_solver` instance (not three), and real
  numbers dropped to **85/220 DSP (39%)** and **59315/53200 LUT (111%)** — DSP now has real
  margin, and LUT overrun is cut from ~3x to ~1.11x but **not fully eliminated**. Trade-off: max
  latency rose to 1594 cycles (~15.9us @ the 10ns/100MHz clock used for this synthesis run, itself
  just a generic default for getting a real number — not a resolution of the separate, explicitly
  off-limits 48MHz/100MHz clock-architecture item above), still far inside the 8.3ms budget.
  **Still open**: closing the remaining ~11% LUT overrun needs either lower-precision `ap_fixed`
  (would need recalibration, not guessed), swapping to the existing unused
  `fpga/external_ip/CORDIC-Unit-main/` CORDIC IP for some of the trig, or further algorithmic
  restructuring — a real architecture choice, not made here.
  Source: `fpga/hls_hardware/InverseKinematics.cpp`, `fpga/hls_hardware/run_csynth.tcl`,
  `fpga/hls_hardware/hls_proj/solution1/syn/report/balance_controller_csynth.rpt` (gitignored
  build artifact — rerun `run_csynth.tcl` to regenerate).

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
- [ ] **`TouchProbe.cpp`'s `SCREEN_HEIGHT_MM` constant fixed to 142.0 in the confirmed-live
  firmware file; NOT yet re-flashed/hardware-verified, historical telemetry NOT rescaled, and 5
  other firmware files define the same constant with unaudited status** `[JETSON-HW to
  re-flash/re-verify]` (flagged 2026-10-14, code fix same day) — was hardcoded to `141.0` since
  this file's first commit, directly scaling every `touch_y_mm` value streamed to the host, vs.
  `CLAUDE.md`'s confirmed 142.0mm source of truth ("never use old values") and
  `docs/PROJECT_LOGBOOK.md`'s independent "CAD Plate vs. Resistive Touchpad Dimensions"
  confirmation. Fixed directly in
  `firmware/stm32_ml_control_and_vision/BallBalancingBot/TouchProbe.cpp` (the one file
  `CLAUDE.md` names as current source of truth) -- a clear, unambiguous constant correction
  against a locked spec value, not a judgment call. **Explicitly NOT done**: (1) re-flashing
  this onto real hardware or re-verifying live (needs the user); (2) retroactively rescaling any
  historically-collected telemetry, or re-running this file's own "FRAME MATCH" reconciliation
  procedure (lines 41-63) -- non-trivial, not attempted; (3) touching the other 5 firmware files
  that also define `SCREEN_HEIGHT_MM` (`firmware/BallBalancingBot/Screen.cpp`,
  `firmware/ExpertEvaluationFirmware/TouchProbe.cpp`,
  `firmware/libraries/VRI_Core/src/Screen.cpp`,
  `firmware/stm32_jetson_remote_control/BallBalancingBot/TouchProbe.cpp`,
  `firmware/stm32_jetson_vla_angle_control/BallBalancingBot/TouchProbe.cpp`) -- deliberately out
  of scope here per `CLAUDE.md`'s own firmware-duplication caution ("never assume a firmware
  directory's purpose from its name alone"); each needs its own live/stale-status check before
  editing, not a blind find-and-replace across all 6. Source:
  `firmware/stm32_ml_control_and_vision/BallBalancingBot/TouchProbe.cpp` lines 34-42, 173;
  `docs/PROJECT_LOGBOOK.md` 14/10/2026.
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
