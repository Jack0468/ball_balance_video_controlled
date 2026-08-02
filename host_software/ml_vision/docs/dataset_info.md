# Dataset Information

This document details the three primary datasets used for training the vision models. Each dataset has distinct characteristics regarding the camera setup, physical angle, and labeling strategy.

---

## Dataset 1: iPhone Images
**Location:** `\host_software\data\02_silver\images_iphone`
**Model Tag:** `_iphone_`

This is the original dataset recorded using an iPhone. It features a specific camera angle that differs from subsequent datasets. The models trained on this dataset represent the first generation of our vision pipeline.

### Coverage Diagnostics
- **Coverage Score**: 99.48% (191/192 cells met the goal of 10 samples per 10x10mm)

![Dataset 1 Coverage Diagnostics](../../data/02_silver/images_iphone/diagnostics/labels_coverage_plot.png)

### iPhone Session Metadata

#### Platform Specifications
- **Dimensions**: Width 187.5 mm, Height 142.0 mm
- **Sensor Calibration**: The physical size of the platform does not precisely match the touch-sensitive boundary of the resistive sensors (i.e., there is a dead zone around the perimeter). To correct this, the center of the ball's coordinates are interpolated and aligned with the physical center of the platform in millimeters.

#### Marker Colors and Positions
| Marker Color | Position (from edges) |
|--------------|-----------------------|
| **Green** | 33mm from left, 26mm from top |
| **Red** | 41mm from right, 53mm from top |
| **Grey** | 69mm from left, 58mm from bottom |
| **Black** | 13mm from right, 8mm from bottom |

#### Sync Anchors & Crop Regions
Below are the synchronization anchors used to align the video frames with the telemetry timestamps.

**Session: 11/07/2026**
- **Video 1**: Crop `[82, 435, 915, 762]`, Frame 0 ➔ Timestamp `1783662365300`
- **Video 2**: Crop `[172, 477, 792, 717]`, Frame 398 ➔ Timestamp `1783663399505` *(Note: Frame index refers to telemetry, not video)*
- **Video 3**: Crop `[130, 485, 812, 710]`, Frame 154 ➔ Timestamp `1783665248539`

**Session: 16/07/2026**
- **Video 4**: Crop `[132, 255, 712, 585]`, Frame 1028 ➔ Timestamp `1784184295854`
- **Video 5**: Crop `[152, 267, 672, 560]`, Frame 209 ➔ Timestamp `1784185262060`

---

## Dataset 2: Hard-coded Angle (Manual Labels)
**Location:** `host_software\data\02_silver\session_20260728_102908`
**Model Tag:** `_0728_`

This dataset introduces a **new camera** at a **hard-coded angle** where the platform and camera are fixed in a set place. The labels for this dataset were manually annotated (or pipelined from earlier models). This dataset was the foundation for some of our most successful intermediate models.

### Coverage Diagnostics
- **Coverage Score**: 88.02% (169/192 cells met the goal of 10 samples per 10x10mm)

![Dataset 2 Coverage Diagnostics](../../data/02_silver/session_20260728_102908/diagnostics/labels_coverage_plot.png)

---

## Dataset 3: Hard-coded Angle (Aruco Markers)
**Location:** `host_software\data\02_silver\session_20260730_174916`
**Model Tag:** `_0730_`

This dataset is taken from the **same hard-coded angle** as Dataset 2, but the platform features a different piece of paper with Aruco markers. We used Aruco marker tracking to precisely determine the homography and location of the platform.

### Coverage Diagnostics
- **Coverage Score**: 85.42% (164/192 cells met the goal of 10 samples per 10x10mm)

![Dataset 3 Coverage Diagnostics](../../data/02_silver/session_20260730_174916/diagnostics/cnn_sequential_features_coverage_plot.png)
