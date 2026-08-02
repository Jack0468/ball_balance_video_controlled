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
- **Process**: We grid the platform into 5mm x 5mm cells, calculate frequencies, and aggressively downsample redundant overlapping coordinates.
- **Output**: `labels_normalized.csv` (A perfectly balanced dataset).

---

## Dataset Loading Strategy
- **ResNet18 (Regression)**: Fed `labels_normalized.csv` directly. Because we dropped frozen/empty frames, it only ever trains on clean, valid coordinates.
- **YOLOv8 (Object Detection)**: Fed the raw `labels_sequential.csv` (which includes missing ball frames). A script (`generate_unified_pose_dataset.py`) projects the physical telemetry into YOLO bounding boxes. If the ball is missing, it omits the class from the label `.txt`, elegantly teaching YOLO what an empty board looks like.
