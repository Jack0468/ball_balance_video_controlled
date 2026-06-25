# Realtime Visual Pipeline Features

This document outlines the core features and components of our realtime ML vision pipeline, which handles live webcam capture, platform detection, and ball tracking.

## Overview
The pipeline consists of two main stages:
1. **Preprocessing**: Automatically detecting the balancing platform and establishing a top-down, 2D normalized perspective.
2. **Inference**: Passing the normalized image into a trained model (YOLO or Classical CV) to track the exact coordinates of the ball on the platform.

---

## 1. Preprocessor (`preprocessor.py`)

The preprocessor is responsible for establishing a stable coordinate space for the inference model. It handles lens undistortion, exposure adjustment, and perspective transformations. 

### Key Features & Improvements:

#### HSV Color Masking (Replaces Canny)
The pipeline now uses an HSV color filter (specifically tuned for low-saturation, high-value colors like White/Grey) to isolate the balancing platform from background clutter. This approach is highly robust against patterned backgrounds and completely bypasses the artificial edges created by the Canny Edge Detector. A binary mask is created where only the platform pixels are white.

#### Morphological Gap Bridging
Following the color masking, a morphological closing operation (`cv2.morphologyEx` with `cv2.MORPH_CLOSE`) using a 7x7 rectangular kernel is applied. This removes salt-and-pepper noise and bridges tiny gaps, increasing the probability of extracting a single, contiguous polygon for the platform.

#### Robust Platform Extraction (Direct Contours)
Instead of running edge detection, the pipeline directly calls `cv2.findContours` on the binary HSV mask. It searches the top 5 largest shapes and applies several heuristics (must be 4-sided, convex, not touch the image borders, and have low internal color variance) to lock onto the platform. If the edge is slightly occluded, it falls back to `cv2.minAreaRect`, which guarantees that the system still bounds the largest contour with exactly 4 corners, preventing complete pipeline failure.

#### Temporal Smoothing (EMA)
To stabilize the generated 2D plane and reduce bounding box jitter from frame to frame, the 4 extracted platform corners are smoothed using an Exponential Moving Average (EMA). 
- An `alpha` of `0.2` is applied.
- Smoothing only occurs if the corners haven't jumped more than a 150-pixel distance threshold, ensuring the camera can still recover tracking if completely moved.
- This creates an incredibly stable top-down feed for downstream models.

---

## 2. Inference Models

### YOLO Inference (`test_yolo_tracking.py`)
Currently, the pipeline uses a pre-trained `YOLOv8n` model. 
- It tracks class `32` (Sports Ball).
- It consumes the warped 2D plane produced by the Preprocessor.

### Classical CV Model (`classical_cv_model.py`)
As a baseline or fallback, a classical HSV color-masking and Hough Circle model is implemented. This relies on strict thresholding but is exceptionally lightweight.

---

## Usage

You can test the realtime end-to-end integration by running:
```bash
python ./scripts/realtime_pipeline_test.py
```
- Accepts `--headless` to skip CV2 rendering for pure latency benchmarks.
- Fallbacks (`--fallback`) can be specified to route the raw camera feed to YOLO if the perspective transformation drops a frame.
