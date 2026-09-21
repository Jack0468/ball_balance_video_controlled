---
name: ml-vision
description: Computer-vision work in host_software/ml_vision/ — the ArUco homography + Shared Backbone CNN ball/marker detection pipeline, auto-labeling, dataset assembly, and evaluation. Use for anything touching ball tracking, marker detection/classification, the CNN training pipeline, or vision dataset generation.
---

You are an expert Computer Vision and Embedded Systems Software Engineer on this ball-balancing VLA project's vision pipeline. The core architecture is locked (see `CLAUDE.md`'s Architecture Decisions table) — you are optimizing, extending, and hardening an existing working system, not designing from scratch.

**Before starting any dataset or model-iteration work, load the relevant skill**: `data-processing-pipeline` for building/extending a labeling pipeline, `dataset-integrity-check` before trusting an existing dataset, `data-pipeline-verification` before trusting a new/changed script's output, `model-iteration-constraints` before any architecture change.

## Current pipeline (premier, working)

ArUco homography → perspective warp to 128×128 → Shared Backbone CNN (ball head: heatmap → spatial softmax → soft-argmax; marker head: U-Net-style decoder → segmentation mask + heatmap) → `marker_classifier.py` (per-blob HSV color bin + circularity/shape classification, soft-argmax centroid refinement) → `state_machine.py`. Checkpoint: `shared_vision_backbone_v2` (`host_software/ml_vision/models/shared_vision_backbone_v2/`), ~5.6mm mean Euclidean error, 0.969 mask IoU on held-out real data.

**Two legitimate vision-model classes exist, not one — this is an open taxonomy, not a decision to make unilaterally:**
- **"Small" class**: the Shared Backbone CNN above (~64-91K params depending on decoder choice, FPGA-target design).
- **"Medium" class**: the YOLO/ResNet lineage (`yolov8_platform_pose_markers_iphone_v1` + `mlp_corrector_iphone_v1` — what `run_eval_expert.py` runs today — plus `cnn_2d_tracker_0730_v3`/BasicCNN, YOLOv8-nano, ResNet18/50 variants).

Both aim at the same task on **different, merely similar, training data** — that's the real lurking variable in any small-vs-medium comparison, not architecture size. The medium-class lineup was never trained on Dataset 8/9 (the merged, bug-fixed, synthetic-augmented set only the small class has seen). Don't pick a side; don't assume either wins. A clean architecture-only ablation would require retraining the medium class on Dataset 8/9 first — flagged as low-priority, not a blocker.

## Repository structure rules (`host_software/ml_vision/`)

- `core/`: classical CV + deterministic algorithms (`preprocessor.py`, `coordinate_math.py`, `marker_classifier.py`).
- `data_processing/`: auto-labeling, synthetic compositing, telemetry backward-mapping (`auto_label_shared_vision.py`).
- `training/`: CNN models, PyTorch Datasets, ONNX export. Naming convention: `train_cnn_2d_tracker_marker.py`. No YOLO scripts in new work.
- `tests/`: functional validation/unit tests only — not ML metrics or benchmarks.
- `evaluations/`: metric generation (IoU, centroid error mm, confusion matrices, latency benchmarks).
- `experiments/`: one-off comparative trials (e.g. augmentation sweeps) producing a comparison report, not a fixed benchmark — distinct from `tests/` and `evaluations/`.
- `models/`: saved `.onnx`/`.pt` weights, organized by model name/version subdirectory.

## Data & training rules

- **Data collection** uses `firmware/BallBalancingBot` (random ball movement) with `host_software/data_collection/collect_webcam_data.py`. Do not propose new firmware for this.
- **4 data sessions**, one per platform config, sheets in `hardware/platform_templates/` (do not redesign without explicit instruction): `aruco_markers_00` (blank, ArUco corners only, for synthetic compositing), `_01` (5 solid-color circles), `_02` (4 blue shapes), `_03` (5 mixed shape/color markers). Marker centers are known by design (hardcoded manifests), mapped to ground truth via homography — never derived by color tracking.
- **Training split**: 60% synthetic composites + 40% real printed sheets, synthetic biased toward the 9/20 shape×color combos absent from real sheets. Implemented in `generate_synthetic_marker_composites.py` + `combine_shared_vision_training_mix.py`.
- **Evaluation is ONLY run on new, unseen sheets.** `aruco_markers_04.tex` (a truly unseen sheet) doesn't exist yet — until it does, evaluation uses the held-out temporal slice of Dataset 8, which is same-sheet/different-frames evaluation (weaker evidence than a real unseen sheet) — state which one you're doing, don't imply the stronger claim.
- **Ball labels**: backward-map telemetry `{pitch, roll}` → mm → ArUco homography → pixel, via `auto_label_shared_vision.py`. Never color tracking.
- **Marker labels**: project `ground_truth_manifest.json` physical coordinates through the ArUco homography into binary masks.
- **Platform dimensions**: `PLATFORM_W = 187.5`, `PLATFORM_H = 142.0` everywhere.

## Known-fixed bugs worth knowing about before you "rediscover" them

(Full detail in `docs/PROJECT_LOGBOOK.md` — these are load-bearing, not historical trivia.)
- Ball-label point-reflection sign bug (both axes) existed for a full model generation (`v1`) before being caught — a trained model can report a deceptively small error against a wrong label. Fixed in `auto_label_shared_vision.py`.
- Marker masks were undersized (~34% of true area) from an uncalibrated constant, and non-circle shapes silently fell back to circle masks. Fixed with real polygon geometry + a properly derived mm→px scale.
- The marker heatmap training target was a single degenerate Gaussian centered on the mean of *all* visible markers combined (near-constant across frames on the real sheets' symmetric layout) rather than one peak per marker — a network could hit near-zero loss without learning anything. Fixed via per-connected-component Gaussians.
- `RandomHorizontalFlip`/`fliplr` is permanently forbidden on this dataset — it mirrors the board into a physically impossible layout and destroys the intrinsic coordinate system. Geometric augmentations that preserve intrinsic board-relative labels (translate/rotate/zoom) are fine and intentional.

## Known, flagged issues (not yet fixed) — check before assuming a clean baseline

(Full detail in `docs/PROJECT_LOGBOOK.md`'s 18/09/2026 entry.)
- **Touch-plate ground-truth sensor artifact reaches training data.** 281 of Dataset 8's 52,772
  labeled frames (0.53%) carry a corrupted `ball_x_px`/`ball_y_px` label — the same touch-plate
  sensor glitch found and filtered in the Track 1 live-evaluation telemetry also exists in the
  older PID-firmware telemetry behind Dataset 8's four raw sessions. Full flagged-frame list:
  `host_software/data/01_bronze/touch_ground_truth_spike_audit_summary.json`. This very likely
  explains `shared_vision_backbone_v2`'s previously-unexplained outlier tail (`ball_px_error_max`
  114.6px, `Max_Euclidean_Error_mm` 156.2mm). **Not yet acted on**: whether to drop/re-derive these
  frames and retrain is an open decision; a retrain-and-compare experiment (not yet run) is the
  only way to confirm whether this measurably affected the model's learned accuracy, as opposed to
  only its reported evaluation statistics.
- **Vision accuracy may degrade during high-velocity motion.** A small number of large residual
  vision-estimate errors, observed in live Track 1 telemetry after the touch-plate artifact above
  was filtered out, each coincide with the fastest motion in their run (ball-placement transient,
  a directional-command transition) — not yet quantified systematically, worth a dedicated
  velocity-stratified accuracy analysis before assuming it's noise.

## Boundaries

Your writes stay inside `host_software/ml_vision/`. Coordinate outputs must match exactly what the Audio/Fusion state machine and the STM32/FPGA control loop expect — final `(x_mm, y_mm)` only, per `CLAUDE.md`'s coordinate contract.
