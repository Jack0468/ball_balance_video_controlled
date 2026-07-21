# ML Vision Data Pipeline

This document explains the end-to-end data processing pipeline used to generate and clean datasets for training the ML Vision models (both ResNet and YOLO architectures) for the VRI 2026 ball-balancing platform.

## Overview

The core challenge of our data pipeline is combining high-frequency hardware telemetry (100Hz+) from the resistive touchscreen with variable-framerate (VFR) video (30fps) from an external camera. Once synchronized, the data must be rigorously cleaned to prevent the models from learning noisy/phantom data, and then normalized to prevent spatial bias.

The pipeline processes data in three distinct stages:
1. **Synchronization**: Aligning video frames with telemetry.
2. **Sequential Cleaning**: Removing duplicates and hardware debouncing artifacts.
3. **Spatial Normalization**: Downsampling to ensure balanced board coverage.

---

## Stage 1: Synchronization (`sync_data.py`)

The first step in dataset creation is running `host_software/ml_vision/data_processing/sync_data.py`. 

- **Input:** Raw `.MOV` video file and raw `telemetry.csv` (100Hz+).
- **Process:** It reads the exact presentation timestamp of every single video frame (to account for smartphone VFR) and performs a binary search to find the closest matching telemetry row based on the synchronized `host_timestamp_ms`.
- **Output:** `labels.csv` (Contains one row for every video frame, but includes duplicate telemetry).

---

## Stage 2: Sequential Cleaning (`clean_sequential_dataset.py`)

Because the telemetry runs much faster than the camera, the initial `labels.csv` is extremely noisy. `host_software/ml_vision/data_processing/clean_sequential_dataset.py` fixes this.

1. **Deduplication:** It first drops duplicate image rows, ensuring there is exactly 1 row per physical video frame (e.g., stripping 240,000 raw rows down to ~56,000 unique frames).
2. **Frozen Frame Filtering:** The platform's resistive touchscreen uses a 1.5-second debouncer to gracefully handle the ball bouncing or falling off. During this time, the `(touch_x, touch_y)` coordinates are "frozen" on the exact same value. If the ML model is trained on these frozen frames, it will be heavily penalized (up to 147mm Max Error) for not guessing the phantom edge coordinate when staring at an empty board. The script detects and deletes any rows where the physical coordinate does not fluctuate by at least 0.1mm (ADC noise).
- **Output:** `labels_sequential.csv` (A perfectly clean, chronologically ordered dataset).

> **Note:** To visually verify this cleaned dataset, you can run `play_dataset.py` to generate an MP4 showing the telemetry overlaid on the video frames.

---

## Stage 3: Spatial Normalization (`normalize_spatial_density.py`)

During raw data collection, the ball naturally spends a disproportionate amount of time near the center of the board. If a CNN is trained on this, it will become biased and lazily guess the center.

- **Process:** `host_software/ml_vision/data_processing/normalize_spatial_density.py` analyzes `labels_sequential.csv`. It grids the platform into 5mm x 5mm cells, calculates the frequency of the majority, and then aggressively downsamples any heavily overlapping coordinates (like the center).
- **Output:** `labels_normalized.csv` (A perfectly balanced dataset, e.g., reduced from ~56,000 to ~24,000 highly valuable, non-redundant training frames).

---

## Dataset Loading Strategy

Depending on which model architecture is being trained, the dataset is loaded differently:

### 1. ResNet18 (Expert Tracker)
ResNet is a Regression model. It *must* output an `(X, Y)` coordinate, meaning it has no concept of a "missing ball." 
- **Training Script:** `train_expert_tracker.py`
- **Data Source:** It is fed `labels_normalized.csv` directly via `ball_dataset.py`. Because we dropped all frozen/empty frames in Stage 2, ResNet only ever trains on clean, valid coordinates.

### 2. YOLOv8 (Future Realtime Architecture)
YOLO is an Object Detection model. It handles missing balls natively by simply predicting zero bounding boxes.
- **Generator Script:** `generate_unified_pose_dataset.py`
- **Data Source:** It takes the cleaned `labels_sequential.csv` and uses homography to project the physical `(touch_x, touch_y)` telemetry into pixel-space bounding boxes, saving them as YOLO `.txt` files in `02_silver_unified_pose`. If the ball is missing from a frame, it simply omits the ball class from the `.txt` file, elegantly teaching YOLO what an empty board looks like.
