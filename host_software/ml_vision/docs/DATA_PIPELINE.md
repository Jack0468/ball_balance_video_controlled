# ML Vision Data Pipeline

This document explains the end-to-end data processing pipelines used to generate and clean datasets for training the ML Vision models for the VRI 2026 ball-balancing platform.

## Overview
We employ two distinct data collection pipelines:
1. **Pipeline A: Webcam Data Collection (Synchronous / Real-Time)**
2. **Pipeline B: iPhone Data Collection (Legacy / Asynchronous)**

Regardless of how the data is collected and synchronized, all datasets pass through rigorous cleaning and spatial normalization stages before training.

---

## 1. Data Collection & Synchronization

### Pipeline A: Webcam Data Collection (Current)
This is the standard, modern method for generating datasets (Datasets 2, 3, and 4). Because the webcam and the serial telemetry are both processed by the same Python script (`host_software/data_collection/collect_webcam_data.py`), the host PC's clock acts as a unified time source.
- **Data Collection**: The script simultaneously records an `rgb_video.mp4`, logs `telemetry.csv`, and records the exact timestamp of every captured frame in `frame_timestamps.csv`.
- **Synchronization**: `host_software/data_collection/sync_webcam_telemetry.py` interpolates the frame timestamps directly with the telemetry timestamps, completely bypassing Variable Frame Rate (VFR) and clock drift issues. 

> **Historical Note on FPGA Logging**: Initially, there was a plan to use the FPGA to directly collect and timestamp webcam data (documented in `fpga_data_logging_plan.md`). **This was never actually implemented.** We ultimately relied strictly on the host PC and USB webcams for data collection.

### Pipeline B: iPhone Data Collection (Legacy)
This pipeline was used exclusively for **Dataset 1**. iOS devices use Variable Frame Rate (VFR) and are completely detached from the PC's clock.
- **The Problem**: We cannot assume Frame 150 happened exactly at 5.000 seconds, and we don't know exactly when the iPhone started recording relative to the Python telemetry logger.
- **The Solution**: A visual timestamp. The Python logger draws a green Unix timestamp on the laptop screen. We point the iPhone at the screen, note the exact frame index and the visible green timestamp, and use that as our anchor.
- **Synchronization**: `host_software/ml_vision/data_processing/sync_data.py` uses this anchor frame to mathematically align the exact presentation timestamp (`pos_msec`) of every video frame to the closest telemetry row, perfectly negating VFR and clock drift.

---

## 2. Image Preprocessing (`preprocess_dataset.py`)
For iPhone data, we run a preprocessor to apply a crop box (extracting the platform), resize all frames to standard 640x480, and extract the final images ready for ML training. 
*(Webcam data is often collected directly at 640x480 with the crop pre-configured via the camera position).*

---

## 3. Sequential Cleaning (`clean_sequential_dataset.py`)
Because telemetry runs much faster (100Hz+) than the camera (30fps), the raw synchronized labels are noisy.
1. **Deduplication**: Drops duplicate image rows, ensuring exactly 1 row per physical video frame.
2. **Frozen Frame Filtering**: The resistive touchscreen has a 1.5s hardware debouncer when the ball is missing. We detect and delete any frames where the physical coordinate does not fluctuate by at least 0.1mm (ADC noise). This prevents models from learning phantom coordinates.

---

## 4. Spatial Normalization (`normalize_spatial_density.py`)
The ball naturally spends a disproportionate amount of time near the center of the board. If a CNN is trained on this, it becomes biased to guess the center.
- **Process**: We grid the platform into 5mm x 5mm cells, calculate spatial frequencies, and aggressively downsample outlier high-frequency cells (e.g., the center) to match the median frequency of the surrounding board. Note that this caps outlier frequencies, but does not perfectly uniformize or artificially upsample low-frequency regions.
- **Output**: `labels_normalized.csv` (An outlier-clipped dataset).

---

## Dataset Loading Strategy
- **ResNet18 (Regression)**: Fed `labels_normalized.csv` directly. Because we dropped frozen/empty frames, it only ever trains on clean, valid coordinates.
- **YOLOv8 (Object Detection)**: Fed the raw `labels_sequential.csv` (which includes missing ball frames). A script (`generate_unified_pose_dataset.py`) projects the physical telemetry into YOLO bounding boxes. If the ball is missing, it omits the class from the label `.txt`, elegantly teaching YOLO what an empty board looks like.
- **Shared Backbone CNN**: Fed `shared_vision_labels.csv` (produced by `auto_label_shared_vision.py`). Each row points to a warped 128×128 image in `images_cropped/` and its corresponding binary marker mask in `masks/`.

---

## 5. Joint Auto-Labeling (`auto_label_shared_vision.py`)

Generates the final training pairs for the Shared Backbone CNN. Run **after** Steps 3 and 4 (requires `labels_normalized.csv`).

For each retained frame, the script:
1. Detects the 4 corner ArUco markers in the raw camera frame and computes a per-frame homography.
2. **Two-stage warp** (host PC only):
   - Stage 1: Perspective warp the raw frame to a **500×500** intermediate image (`cv2.warpPerspective`). This preserves sub-pixel accuracy during the perspective correction interpolation.
   - Stage 2: Downsample to **128×128** using `cv2.INTER_AREA` (area averaging). Saved to `<session_dir>/images_cropped/`.
3. Reprojects ball telemetry coordinates (mm → camera px → warped 128×128 px) into the final coordinate space.
4. Renders binary segmentation masks for each marker in 128×128 space using the `ground_truth_manifest.json` physical positions. Saved to `<session_dir>/masks/`.
5. Outputs `<session_dir>/shared_vision_labels.csv` with `image_file`, `ball_x_px`, `ball_y_px`, and all original telemetry columns.

---

## FPGA Inference vs. Host PC Training Pipeline Mismatch

> [!IMPORTANT]
> The **500×500 intermediate warp buffer exists only in the training pipeline on the host PC**. It cannot run on the FPGA PL and is explicitly not intended to.

### Why the mismatch exists

| | Host PC (Training) | FPGA PL (Inference) |
|---|---|---|
| Warp strategy | Camera → **500×500** (warpPerspective) → **128×128** (INTER_AREA) | Camera → **128×128** direct streaming warp |
| Reason | Maximize label accuracy; sub-pixel quality during interpolation | 500×500 intermediate frame = ~732 KB — exceeds entire 612.5 KB BRAM budget |
| ArUco detection | Python OpenCV (per-frame) | ARM Cortex-A9 PS (ZedBoard Processing System) |
| Homography apply | Python `cv2.warpPerspective` | FPGA PL streaming bilinear interpolation (no frame buffer) |

### Why this is safe

The two-stage warp and the single-pass warp are **mathematically identical geometric transformations**. The only difference is the interpolation quality of the intermediate step. The 500×500 intermediate produces marginally cleaner training images and more precise mask pixel coordinates, but the FPGA's single-pass warp produces an image that is visually and numerically equivalent. The trained model will generalise correctly to FPGA-warped inference frames because both operations implement the same projective geometry.

### FPGA pipeline (Phase 5)
```
Camera stream (640×480)
    ↓
ARM PS: ArUco detect → compute homography matrix H (cached until camera moves)
    ↓ (H sent over AXI to PL)
FPGA PL: Per-pixel streaming warp → 128×128 (bilinear, no intermediate buffer)
    ↓
FPGA PL: Shared Backbone CNN (70K params, 100% BRAM-resident)
    ↓
Ball (x,y) px  +  Marker binary mask
    ↓
ARM PS: px → mm coordinate conversion → STM32 PID control loop
```

---

## Physical Calibration Notes (2026-08-10)

> [!NOTE]
> These notes were recorded during the first physical data collection session. They document known systematic errors that are acceptable within the current architecture.

### Print Scaling Error

During the first print run of the platform templates (`aruco_markers_00` through `aruco_markers_03`), no ruler was available for precise verification. Post-hoc measurement revealed:

- **Horizontal axis error:** approximately **≤ 10mm** (print was slightly undersized)
- **Vertical axis error:** not yet measured precisely
- **Session-to-session consistency:** All 4 sessions were printed and collected on the same day using the same printer settings. The **relative scale between platforms is consistent** — the error is a fixed systematic offset, not random per-sheet variation.

### Implication: Translation Offset Only Matters for `aruco_00`

For sessions using `aruco_markers_01` through `aruco_markers_03`, the ArUco homography origin (top-left of the printed paper) and the physical platform centre are related only by a fixed offset baked into the manifest. Even with a ≤10mm print scale error, the relationship between the homography coordinate frame and the physical touchpad telemetry is **consistent across sessions** — the CNN will not see inconsistent coordinate systems between sheets.

For `aruco_markers_00` (blank platform, used for synthetic compositing), the offset between the ArUco homography (0,0) and the physical centre of the touchpad is estimated at **< 5mm** given the observed scale error.

### MLP Corrector Role

The planned small MLP corrector (`9 → 32 → 32 → 3`, ~1.5K params) is expected to absorb the residual scale and translation offset between:
- The ArUco-homography-derived pixel coordinate space
- The physical mm coordinate space expected by the STM32 PID loop

Because the error is systematic and consistent across sessions, the MLP will be able to learn a stable correction function from the training data without needing perfectly calibrated prints.

### Recommended Next Step

Once a ruler is available, re-measure the printed platform to confirm the exact scale factor. Update `PAPER_W` and `PAPER_H` constants in `auto_label_shared_vision.py` if the error is larger than 5mm in either axis.

---

## Missing Padding in `auto_label_shared_vision.py`

> [!WARNING]
> The current `warp_to_platform()` function does **not** pad the source camera frame before warping. If any platform corner falls outside the 640×480 camera frame boundary, `cv2.warpPerspective` will silently fill those pixels with black rather than raising an error.

The legacy pipeline (`generate_aruco_cropped_dataset.py`, line 11) used `CROP_PAD = 20px` — it first found the platform boundary in pixel space, added a 20px border crop, and warped from that crop. This ensured the full platform was always visible in the warp input.

**Current workaround:** The camera is physically positioned to capture the full platform in frame, so this is unlikely to cause black regions in practice. However, if the camera is bumped or the platform is not perfectly centred in frame, silent black edge regions will appear in `images_cropped/` without any warning.

**Planned fix:** Add `cv2.copyMakeBorder` padding to the source frame in `warp_to_platform()` before calling `cv2.warpPerspective`, and offset the `src_px` corner coordinates accordingly.
