# Does the ball occlude the real black marker on the go_black failure frames? (offline check, 2026-10-07)

Offline analysis only, Windows machine. Read-only: no files changed under `host_software/ml_vision/`,
`host_software/data/`, or anywhere else. No commits.

## Question

`host_software/ml_jetson_vla/reports/black_marker_aruco_check_2026_10_07.md` found Moondream2
`:point` grounds "black marker" onto an ArUco calibration square on all 13 `go_black` frames, but
left one alternative unresolved: the black target is small and the ball may simply be **occluding**
it in these frames, which would independently explain a fallback to the next most salient object.
That was never measured. This report measures it, directly, on the real frames.

## Verdict

**Occlusion is a real, substantial factor in most (12/13, 92%) of the 13 frames at realistic/generous
ball-radius assumptions (10-15mm radius) — but it is refuted as the *sole* explanation by one clean
counter-example where the ball is durably, verifiably away from the marker and Moondream still
answered on an ArUco tag.** At the most conservative ball-radius assumption (5mm) only about half
(7/13, 54%) of the frames show overlap, so the strength of "occlusion explains most of it" is
sensitive to the real ball radius, which this project does not have on record (same caveat the
`ball_over_marker` report already carries).

- **Primary, trustworthy measurement**: the ball's real pixel/mm position (telemetry `touch_x`/
  `touch_y`, projected through each frame's own ArUco homography) versus the real black marker's
  manifest-anchored position (derived independently per session from where that session's own
  `go_black` *commanded target* telemetry converges — not from the classical-CV detector, which
  turned out to be unreliable here, see below, and not forced to match a specific manifest file's
  claimed shape).
- Across the 13 frames: ball-to-marker-centre distance ranges **5.1 to 26.3mm** (median ~12.7mm).
  Modelling the marker as an 8mm-radius circle (`render_marker_mask()`'s own convention for
  `size_mm` on a circle feature) and sweeping ball radius 5/10/15mm (same three radii
  `ball_over_marker`'s own footprint test used):
  - ball radius 5mm: **7/13 overlap**
  - ball radius 10mm: **12/13 overlap**
  - ball radius 15mm: **12/13 overlap** (same 12 — one frame never overlaps at any radius tested)
- **The one non-overlapping frame**: `session_jetson_track4_20260915_163702`, `frame_index 2287`
  — ball-to-marker 26.3mm, edge-to-edge gap 18.3mm (no overlap even at the 15mm ball radius).
  Checked this isn't a settling-transition artifact: the `go_black` run in that session spans
  frames 1560-2471 (911 frames), and frame 2287 sits in the middle of a long stretch (frames
  ~2270-2300 inspected) where `touch_x/touch_y` oscillates around (20-27, -22 to -30) while the
  target sits near (0, -30) — the ball genuinely never settles near the marker for hundreds of
  frames here, not a brief in-flight moment. The companion report's own numbers already show
  Moondream answered within ~3mm of an ArUco tag on this exact frame. So on this frame the real
  target was visible and the model still chose a tag — occlusion cannot be the explanation there.
- **Detector reliability (read this before trusting any single-frame "detected" number)**: the
  project's classical HSV blob tracker (`MarkerTracker.find_targets()`,
  `host_software/ml_vision/core/marker_tracker.py`) found **zero** black blobs across all 13
  frames — its hardcoded black bound (`V<=60`) is stale; `marker_classifier.py`'s own 2026-09-15
  recalibration note already measured the real printed black marker at `V~110-140`. Built a
  documented fallback (see "Detector" section below) reusing `marker_classifier.py`'s *current*
  calibrated black HSV bin on the ArUco-warped frame. That fallback found *some* dark blob on all
  13 frames, but cross-checked against the independently-derived marker position it was only
  plausibly correct (within ~20mm) on **2/13** frames and landed **28-111mm away** on the other
  **11/13** — visually confirmed in the annotated images to be locking onto an ArUco tag's own
  black square instead (see `black_occlusion_case_detector_fails_153053_f1533.png`). This is a
  real finding in its own right (the project's classical detector reproduces a milder version of
  the same tag-confusion Moondream shows) but it means the fallback detector's output was **not**
  used as the primary distance measurement — doing so would have overstated "no overlap" almost
  everywhere, for the wrong reason (measuring ball-to-ArUco-tag, not ball-to-real-marker distance).

## Data and method

- **The 13 frames**: `instruction == "go_black"` rows across all sessions/variants in
  `host_software/data/arm2_jetson_sweep_results/scorer_format_moondream2_point_jetson_run1.json`
  (the exact path given in the task — matches the companion report's 13-frame count and 7
  sessions). Extracted directly from the JSON (filtered, not re-typed from the prior report's
  prose), one `baseline` variant per frame:

  | session | frame_index |
  |---|---|
  | session_jetson_track4_20260915_151627 | 1531 |
  | session_jetson_track4_20260915_153053 | 1533 |
  | session_jetson_track4_20260915_153247 | 1534 |
  | session_jetson_track4_20260915_160509 | 1695 |
  | session_jetson_track4_20260915_161239 | 3365 |
  | session_jetson_track4_20260915_161239 | 4263 |
  | session_jetson_track4_20260915_163702 | 1711 |
  | session_jetson_track4_20260915_163702 | 2287 |
  | session_jetson_track4_20260915_164115 | 2898 |
  | session_jetson_track4_20260915_164115 | 3198 |
  | session_jetson_track4_20260915_164115 | 3308 |
  | session_jetson_track4_20260915_164115 | 3418 |
  | session_jetson_track4_20260915_164115 | 3528 |

- **Frame decode**: `host_software/data/01_bronze/<session>/rgb_video.mp4`, sequential PyAV decode
  to the exact frame index — same mechanism as
  `host_software/ml_jetson_vla/deployment/score_minimal_baseline_offline.py`'s
  `read_frames_at_indices` (re-implemented as a one-index version in the scratch script; same
  library, same sequential-decode approach, no `cv2.VideoCapture` seeking).
- **Homography**: `estimate_homography_from_aruco()` +
  `load_manifest_full(ground_truth_manifest.json)` from
  `host_software/ml_vision/data_processing/auto_label_shared_vision.py` (read-only reuse, same as
  both prior reports). `ground_truth_manifest.json`'s `aruco_markers` list (the 6 corner/edge
  tags) is identical across the `00`/`01`/`02`/`03` per-sheet manifests, so this is valid
  regardless of which colour-marker sheet a session used.
- **Ball position**: telemetry `touch_x`/`touch_y` (the actual sensed ball position, not
  `target_x`/`target_y`, which is the *commanded* setpoint) from each session's `telemetry.csv`,
  row matched by `frame_index` (confirmed `frame_index` == row index, monotonic, for this data).
  Converted to the manifest mm frame with the same relation `auto_label_shared_vision.py` and the
  scorer use: `mm_x = W/2 - touch_x`, `mm_y = H/2 + touch_y`.
- **Real black-marker position ("known_manifest_mm")**: `hardware/platform_templates/
  aruco_markers_01_manifest.json` and `..._03_manifest.json` both carry a `feature_black_circle`
  (shape `circle`, `size_mm` 8.0, which `auto_label_shared_vision.py`'s `render_marker_mask()`
  docstring confirms is the shape's *radius*, not diameter). I did **not** just pick a manifest
  file by filename/session-naming guess: I derived which one actually applies to each session from
  that session's own `go_black` *commanded-target* telemetry (`true_target_x/y_mm` in the scored
  JSON, converted to manifest mm the same way) — it clusters tightly at `(93.75, 71.0)` for
  sessions 151627/153053/153247/160509/161239 (matches `aruco_markers_03`'s
  `feature_black_circle`) and at `(93.75, 41.0)` for 163702/164115 (matches `aruco_markers_01`'s
  `feature_top_black`). This is itself real evidence, not an assumption: the firmware/host that
  issues `go_black` commands the ball to a coordinate, and that coordinate is reproduced near-
  exactly across every frame of a given session. **Caveat**: visual inspection of the annotated
  frames shows the physically photographed marker layout for the 163702/164115 sessions doesn't
  exactly match `aruco_markers_01`'s "5 solid colour circles" (the frames show 4 circles plus what
  looks like a dark triangle, not 5 circles) — i.e. the real printed sheet used in those sessions
  isn't a byte-for-byte match to any of the 3 manifest JSONs in the repo. The *coordinate* is still
  trustworthy (it's reproduced by the robot's own repeated commanded setpoint, not read off the
  manifest's shape label), and it was cross-checked visually against all 7 annotated images (the
  cyan diamond marker consistently sits immediately next to / under the ball across frames of the
  same session, never off in some unrelated part of the platform) — but the exact printed shape at
  that location should not be taken as "definitely a circle" from this check alone.
- **Detector**: tried `MarkerTracker.find_targets()`
  (`host_software/ml_vision/core/marker_tracker.py`) first, on the frame warped to a top-down
  view via `auto_label_shared_vision.warp_to_platform()` (500x500, for blob-detection fidelity —
  an 8mm marker is only ~5px radius at the CNN's native 128x128). It found 0/13 black blobs
  (stale `V<=60` bound, see Verdict). Neither `MarkerTracker` nor `MarkerClassifier.classify()`
  (`host_software/ml_vision/core/marker_classifier.py`, which needs a CNN mask + heatmap, not a
  raw image) exposes a standalone "find the black blob in this image" call using the project's
  *current* calibrated black bin, so I built the simplest faithful fallback: `cv2.inRange()` with
  `marker_classifier.COLOR_BINS["black"]` (the project's own current, recalibrated threshold — not
  a new one I invented) on the same warped frame, then contour/area/circularity filtering
  structurally modelled on `MarkerTracker`'s own approach (closest-to-expected-marker-area wins,
  not highest circularity, since a partially-occluded blob is exactly what we're testing for).
  This is documented in the scratch script's `detect_black_blob_calibrated()` docstring. Its
  unreliability (11/13 frames land on the wrong dark feature) is reported plainly above, not
  smoothed over.
- **Overlap/footprint test**: same approach `ball_over_marker`'s report/CSVs used (distance from
  ball centre to the nearest point on the marker's own footprint, compared against candidate ball
  radii), adapted for a circular marker footprint instead of `ball_over_marker`'s square ArUco-tag
  footprint: `edge_dist_mm = max(0, euclidean_dist(ball, marker_centre) - marker_radius_mm)`,
  overlap at ball radius `r` iff `edge_dist_mm <= r`. Same three radii (5/10/15mm) as that report.
- **touch_stale / touch_glitch sanity check**: checked each frame's `touch_x`/`touch_y` against
  the previous telemetry row (exact-repeat `touch_stale` flag, per the existing project
  convention) and against the local 7-frame median (today's newly-flagged touch-plate glitch
  pattern, logbook 2026-10-07). Only 1 of the 13 frames is `touch_stale`
  (`session_jetson_track4_20260915_153247`, frame 1534 — a 1-row repeat, not a multi-frame run);
  **none** show the >30mm jump pattern of today's newly-found touch-plate glitch. The ball position
  used here is not corrupted by either known telemetry artifact.

## Per-frame numbers

| session (suffix) | frame | touch_stale | ball-to-known-marker (mm) | edge dist (mm) | overlap @5mm | @10mm | @15mm | detector-vs-known (mm) |
|---|---|---|---|---|---|---|---|---|
| 151627 | 1531 | False | 11.5 | 3.5 | yes | yes | yes | 5.0 |
| 153053 | 1533 | False | 11.4 | 3.4 | yes | yes | yes | 101.3 |
| 153247 | 1534 | **True** | 15.4 | 7.4 | no | yes | yes | 81.8 |
| 160509 | 1695 | False | 9.8 | 1.8 | yes | yes | yes | 19.9 |
| 161239 | 3365 | False | 8.4 | 0.4 | yes | yes | yes | 81.6 |
| 161239 | 4263 | False | 17.1 | 9.1 | no | yes | yes | 28.8 |
| 163702 | 1711 | False | 17.2 | 9.2 | no | yes | yes | 87.2 |
| **163702** | **2287** | False | **26.3** | **18.3** | **no** | **no** | **no** | 87.2 |
| 164115 | 2898 | False | 14.8 | 6.8 | no | yes | yes | 88.1 |
| 164115 | 3198 | False | 10.6 | 2.6 | yes | yes | yes | 87.3 |
| 164115 | 3308 | False | 5.1 | 0.0 | yes | yes | yes | 87.1 |
| 164115 | 3418 | False | 12.7 | 4.7 | yes | yes | yes | 87.1 |
| 164115 | 3528 | False | 13.4 | 5.4 | no | yes | yes | 45.8 |

"detector-vs-known" = distance between the calibrated-bin fallback detector's output and the
independently-derived known marker position; large values (28-111mm) mean the detector landed on
something else (an ArUco tag, in every annotated image checked), not the real marker.

## Annotated images (4 of 13, representative; full 13 in the reproduction script's output dir)

- `black_occlusion_case_overlap_151627_f1531.png` — sheet with the black marker at the platform's
  geometric centre; ball sits almost exactly on it (11.5mm, overlaps at every radius tested).
  Ball drawn as green cross + three yellow rings (5/10/15mm radii, raw-pixel scale estimated
  locally from the homography). Blue boxes = the 6 ArUco tags. Cyan diamond = the known marker
  position used for the primary measurement. Red circle = the fallback detector's output (lands
  correctly here, 5.0mm from the known position — one of only 2/13 frames where it does).
- `black_occlusion_case_strong_overlap_164115_f3308.png` — closest overlap in the set (5.1mm,
  edge distance ~0, ball centre essentially inside the marker's own 8mm radius).
- `black_occlusion_case_NO_overlap_163702_f2287.png` — **the one counter-example**: ball and known
  marker position (cyan diamond) sit close together in the frame but with a visible, real gap;
  computed 26.3mm centre distance / 18.3mm edge gap, no overlap at any radius tested, confirmed by
  the telemetry-trajectory check above.
- `black_occlusion_case_detector_fails_153053_f1533.png` — illustrates the detector reliability
  problem: red circle (fallback detector output) sits on the **top-left ArUco tag**, 101.3mm from
  the real marker position, while the ball and cyan diamond (independently-derived true marker
  position) sit together correctly in the platform's centre. The project's own classical detector
  reproduces a milder version of the same ArUco-tag confusion documented for Moondream2.

## Reproduction

Scratch script (not in the repo): `analyze.py` in
`C:/Users/Admin/AppData/Local/Temp/claude/c--Users-Admin-Documents-Windows-codespace-VRI-2026/facca623-bcfc-421e-90d5-797d32ca56bf/scratchpad/black_marker_occlusion/`
— writes `per_frame_results.json` and all 13 annotated PNGs there. Uses the pinned interpreter
`C:/Users/Admin/.conda/envs/ball_balance_env/python.exe`.

## Caveats

1. **Ball radius is not on record for this rig.** The 5/10/15mm sweep (matching `ball_over_marker`'s
   own choice) is a stand-in for an unmeasured real value — the "12/13 overlap" headline number
   depends on the true radius being closer to 10-15mm than 5mm.
2. **The real printed sheet for 2 of the 7 sessions doesn't exactly match any manifest JSON in the
   repo** (see "Real black-marker position" above) — the *coordinate* used is trustworthy (derived
   from the robot's own repeated commanded setpoint, cross-checked visually), but the marker's true
   shape there may not be a circle, and the 8mm circle-footprint model is an approximation for
   whatever it actually is.
3. **Small sample, single counter-example.** 13 frames, 7 sessions, 1 run. The one non-overlapping
   frame is real and checked two independent ways (telemetry trajectory + visual), but it is one
   frame — it is enough to refute "occlusion is the sole explanation," not enough to quantify how
   much of the remaining, non-occlusion effect size is real tag-confusion versus something else.
4. **Detector unreliability is the headline finding for the classical-CV path, separate from the
   VLM question** — flagged plainly above, not treated as resolved. A real "does our own vision
   pipeline also confuse this marker with an ArUco tag" question is now open and quantifiable
   (11/13 wrong here), but this check did not try to fix or retune the detector, which is out of
   this task's scope.
