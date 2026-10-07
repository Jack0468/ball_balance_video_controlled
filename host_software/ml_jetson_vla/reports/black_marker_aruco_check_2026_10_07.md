# Black-command mis-grounding vs. ArUco calibration markers (offline check, 2026-10-07)

Offline analysis only, Windows machine. No Jetson access, no changes under `host_software/data/`, no Dockerfile or commit changes.

## Question

`docs/ARM2_MINIMAL_BASELINE_MULTI_CANDIDATE.md` section 13 reports Moondream2 `:point` at 0/13 hits on `go_black` frames while parsing all 13. Hypothesis: the model grounds "black marker" onto one of the black-and-white ArUco calibration squares rather than the black colour marker. Section 10.1 also records Qwen's black command as its weakest (38% hit). Test whether the hypothesis holds, for both models.

## Verdict

**Supported for Moondream2 `:point`. Refuted as the explanation for Qwen's black weakness.**

- All 13 Moondream black answers land within 0.4 to 3.0 mm of a manifest ArUco centre (nearest marker id 2 or 3 in most frames, id 0 once). Those same answers are 99 to 124 mm from the telemetry truth. Every one of those 13 is also inside the 22.5 mm tag footprint (Chebyshev distance to the centre at most 11.25 mm).
- The true black target is 80 to 87 mm from the nearest ArUco centre in every black frame, so the ArUco match is not a truth-side coincidence.
- Qwen's 8 black misses are not near any ArUco marker (nearest 33.6 to 86.1 mm). Six of the eight are just misses at 20.5 to 23.1 mm, which is a localisation tail, not a tag confusion.
- Caveat that is not resolved here: in the frames I looked at, the black marker is a small dark triangle lying under the ball at the commanded position. The ball may be occluding the target, so the model may be choosing the ArUco because the real target is hidden. That is an occlusion explanation, and the data here cannot separate it from "confused with a black-and-white square". See the Caveats section.

## Data and method

- Predictions: `host_software/data/arm2_jetson_sweep_results4/scorer_format_moondream2_point_jetson_run1.json` (variant `baseline`, the only one) and `scorer_format_qwen2_5_vl_3b_jetson_run1.json` (variant `baseline`). Both contain the same 60 (session, frame_index) pairs. 51 Moondream frames parsed (9 unparsed, all red/yellow). All 60 Qwen frames parsed.
- Frames: decoded from `host_software/data/01_bronze/<session>/rgb_video.mp4` with the scorer's `read_frames_at_indices` (PyAV sequential decode). All 60 decoded.
- Homography: per frame, `estimate_homography_from_aruco` from `host_software/ml_vision/data_processing/auto_label_shared_vision.py`, with the manifest lookup from `load_manifest_full`. Every frame had a homography and all six manifest markers were detected in every black frame.
- Coordinate handling: the predicted raw pixel (`target_point_raw_px`, which already accounts for the backend's coordinate space) goes through the inverse of the per-frame homography (the scorer's `raw_px_to_mm`) into the manifest frame. Truth goes through the scorer's `touch_frame_to_manifest_mm` (`W/2 - x`, `y - H/2`). Recomputed predictions match the stored `pred_target_*_mm` to within 0.21 mm across 111 parsed predictions, and recomputed error matches the stored `error_mm` to within 0.19 mm.
- ArUco reference: the six `aruco_markers` centres in `hardware/platform_templates/ground_truth_manifest.json` (manifest frame, top-left origin). Nearest-marker distance is the Euclidean distance in mm to the nearest of those six centres. "Near" means at most 20 mm, matching the hit tolerance. The footprint check uses Chebyshev distance at most 11.25 mm (half of the 22.5 mm tag).
- Hit definition: error at most 20 mm, as in the scorer.

## Numbers

### Go_black (13 frames, 7 sessions)

| model | hits | misses | misses within 20 mm of an ArUco | misses beyond 20 mm of any ArUco | misses inside tag footprint | median miss error (mm) |
|---|---|---|---|---|---|---|
| Moondream2 `:point` | 0/13 | 13 | **13 (100%)** | 0 | 13 | 119.7 |
| Qwen2.5-VL-3B baseline | 5/13 | 8 | **0 (0%)** | 8 | 0 | 22.9 |

Moondream nearest-ArUco distances for its 13 black answers: 0.4, 0.6, 1.0, 1.0, 1.3, 1.3, 1.5, 1.5, 1.6, 1.8, 2.4, 3.0, 3.0 mm (all 13 within 3.0 mm).

Qwen black misses: errors 131.3, 22.0, 23.1, 21.4, 64.9, 20.5, 22.8, 153.1 mm. Nearest ArUco for those eight: 84.9, 61.8, 66.4, 70.7, 33.6, 69.7, 66.5, 86.1 mm.

### Control, go_red / go_green / go_yellow

| model | frames | parsed | hits | misses | misses within 20 mm of an ArUco | misses beyond 20 mm | median miss error (mm) |
|---|---|---|---|---|---|---|---|
| Moondream2 `:point` | 47 | 38 | 35 | 3 | 0 | 3 | 29.2 |
| Qwen2.5-VL-3B baseline | 47 | 47 | 36 | 11 | 2 | 9 | 41.3 |

The two near-ArUco Qwen control misses are one go_red (error 97.8 mm, 16.3 mm from ArUco 5) and one go_green (error 81.9 mm, nearest ArUco within 20 mm). So Qwen does sometimes land on a tag, but rarely and not specifically on black: 0 of 8 black misses versus 2 of 11 control misses.

### Pooled

| model | misses | misses within 20 mm of ArUco |
|---|---|---|
| Moondream2 `:point` (all 51 parsed) | 16 | 13 (81%) |
| Qwen2.5-VL-3B baseline (all 60) | 19 | 2 (11%) |

Per-command, per-model JSON: `summary.json` and `per_frame.json` in the scratchpad. The scratch folder is the agent scratchpad, not the repo.

## Annotated images

Each image shows the raw frame, blue boxes (about 22 px) at the six ArUco centres (projected through the per-frame homography), a green cross at the telemetry truth, and a red dot at the model's point. The banner gives the error and nearest-ArUco distance.

- `host_software/ml_jetson_vla/reports/black_marker_case_A_black_moondream_on_aruco.png`: go_black, Moondream point on ArUco 3 (error 123.6 mm, 3.0 mm from the tag). This is the hypothesis case.
- `host_software/ml_jetson_vla/reports/black_marker_case_B_black_qwen_large_no_aruco.png`: go_black, Qwen baseline error 153.1 mm, nearest ArUco 86.1 mm. This is a large miss with no ArUco involved.
- `host_software/ml_jetson_vla/reports/black_marker_case_C_control_qwen_miss_near_aruco.png`: go_red, Qwen baseline error 97.8 mm, 16.3 mm from ArUco 5. Control case showing that Qwen's tag confusion is not black-only.
- `host_software/ml_jetson_vla/reports/black_marker_case_D_control_moondream_miss_far.png`: go_red, Moondream error 30.1 mm, nearest ArUco 51.6 mm. Control case for a large-ish miss with no tag involved.

Visual observations from the four images and one extra scratch frame (not in the report folder):
- The colour markers differ in shape and colour between sessions. In session `..._151627` the centre-row dots are round (green, red, blue, yellow). In session `..._161239` they are a green hexagon, a yellow square, a red triangle, and a dark-blue triangle. The black target is that dark triangle.
- In the black frames checked, the ball sits on or next to the black target's position. In case A the cross is under the ball, and in the scratch frame (session `..._161239`, frame 4263, truth y=71 mm in the manifest frame) the ball covers most of the dark triangle.

## Caveats

1. **Occlusion is a confound.** The black marker appears to be partly or wholly under the ball at the commanded position in the frames inspected. The model may be picking the ArUco because the real target is covered. I did not measure ball occlusion across all 13 frames: there is no ball detector in this path, and the four case images plus one scratch frame were the only visual check. The result shows the model prefers a tag to the hidden marker. It does not show the model is confusing black-and-white shapes in general.
2. **Small sample.** 13 black frames from 7 sessions, one run, one prompt (`:point` ignores the prompt). The aggregate hit-rate numbers are from a single sweep, as the doc notes.
3. **20 mm tolerance versus tag size.** A 20 mm radius covers most of a 22.5 mm tag. The footprint check (Chebyshev at most 11.25 mm) gives the same count for Moondream (13/13 inside the footprint), so the result does not depend on the radius choice.
4. **Truth source.** Truth is the telemetry `target_x/target_y` converted to the manifest frame. Colour-marker positions are not in the manifest (`features` is empty). The black truth position is the telemetry setpoint, not an independently measured marker centroid. I did not verify it against a colour-segmentation centroid for the black marker, unlike the 2026-09-19 check for the other colours.
5. **Qwen is not black-specific.** Qwen does land near ArUco markers on colour frames (2 of 11 misses), so tag confusion is a general failure mode for it, just not the reason for its black weakness.

## Reproduction

Scratch scripts (not in the repo): `C:/Users/Admin/AppData/Local/Temp/claude/c--Users-Admin-Documents-Windows-codespace-VRI-2026/facca623-bcfc-421e-90d5-797d32ca56bf/scratchpad/black_marker_check.py` (per-frame analysis, writes `per_frame.json` and `summary.json`) and `draw_cases.py` (renders the four case images). Both use the pinned interpreter `C:/Users/Admin/.conda/envs/ball_balance_env/python.exe`.

## Suggested next step (not run)

To separate occlusion from tag confusion: pick black frames where the ball is provably away from the black marker's position (for example by ball-position telemetry or a colour-segmentation check) and count the same nearest-ArUco statistic. If Moondream still lands on the tags, the confusion is real. If its answers move onto the black marker when unoccluded, the cause is occlusion.
