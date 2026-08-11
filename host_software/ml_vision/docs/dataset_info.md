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
| -------------- | ----------------------- |
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

## Dataset 4: Aruco Markers + Hard coded camera

**Location:** `host_software\data\02_silver\session_20260810_104132`
**Model Tag:** ``

This dataset is taken from the **same hard-coded angle** as Dataset 2 and 3, but the platform features a different piece of paper with Aruco markers. We used Aruco marker tracking to precisely determine the homography and location of the platform.

Part of new set that has the same aruco features. BUT no coloured markers are included.

see aruco_markers_00.tex file for reference

### Processing log

```text
clean_sequential_dataset.py - >
Initial rows: 28794
Removed 0 duplicate image rows.
Removed 6160 frozen telemetry frames (ball off-board or bouncing).
Final Cleaned rows: 22634

Reading dataset: host_software\data\01_bronze\session_20260810_104132\labels_sequential.csv
Total rows: 22634
Grid cell frequencies - min: 1, max: 228, median: 20
Frequency outlier threshold (Q3 + 1.5*IQR): 66.5
Target majority frequency: 20
Rows after normalization: 14840 (dropped 7794)
```

### Coverage Diagnostics

- **Coverage Score**: 99.48% (191/192 cells met the goal of 10 samples per 10x10mm)

![Dataset 4 Coverage Diagnostics](../../data/01_bronze/session_20260810_104132/shared_vision_labels_coverage_plot.png)

## Dataset 5: Aruco Markers + Hard coded camera + coloured circles

**Location:** `host_software\data\02_silver\session_20260810_110239`
**Model Tag:** ``

This dataset is taken from the **same hard-coded angle** as Dataset 2 and 3, but the platform features a different piece of paper with Aruco markers. We used Aruco marker tracking to precisely determine the homography and location of the platform.

Part of new set that has the same aruco features. coloured markers are included as the coloured circles
see aruco_markers_01.tex file for reference

### Processing log

```text
Loading host_software\data\01_bronze\session_20260810_110239\synced_telemetry.csv...
Initial rows: 20850
Removed 0 duplicate image rows.
Removed 2844 frozen telemetry frames (ball off-board or bouncing).
Final Cleaned rows: 18006
Saved cleaned dataset to host_software\data\01_bronze\session_20260810_110239\labels_sequential.csv

Reading dataset: host_software\data\01_bronze\session_20260810_110239\labels_sequential.csv
Total rows: 18006
Grid cell frequencies - min: 1, max: 291, median: 15
Frequency outlier threshold (Q3 + 1.5*IQR): 48.0
Target majority frequency: 15
Rows after normalization: 10992 (dropped 7014)
Saved normalized dataset to: host_software\data\01_bronze\session_20260810_110239\labels_normalized.csv
```

### Coverage Diagnostics

- **Coverage Score**: 100.00% (192/192 cells met the goal of 10 samples per 10x10mm)

![Dataset 5 Coverage Diagnostics](../../data/01_bronze/session_20260810_110239/shared_vision_labels_coverage_plot.png)

## Dataset 6: Aruco Markers + Hard coded camera + Blue Shapes

**Location:** `host_software\data\01_bronze\session_20260810_112047`
**Model Tag:** ``

This dataset is taken from the **same hard-coded angle** as Dataset 2 and 3, but the platform features a different piece of paper with Aruco markers. We used Aruco marker tracking to precisely determine the homography and location of the platform.

Part of new set that has the same aruco features. coloured markers are included as the blue shapes
see aruco_markers_02.tex file for reference

### processing log

```text
Loading .\host_software\data\01_bronze\session_20260810_112047\synced_telemetry.csv...
Initial rows: 26677
Removed 0 duplicate image rows.
Removed 3627 frozen telemetry frames (ball off-board or bouncing).
Final Cleaned rows: 23050
Saved cleaned dataset to .\host_software\data\01_bronze\session_20260810_112047\labels_sequential.csv

Reading dataset: host_software\data\01_bronze\session_20260810_112047\labels_sequential.csv
Total rows: 23050
Grid cell frequencies - min: 1, max: 420, median: 20
Frequency outlier threshold (Q3 + 1.5*IQR): 64.0
Target majority frequency: 19
Rows after normalization: 14118 (dropped 8932)
Saved normalized dataset to: host_software\data\01_bronze\session_20260810_112047\labels_normalized.csv
```

### Coverage Diagnostics

- **Coverage Score**: 100.00% (192/192 cells met the goal of 10 samples per 10x10mm)

![Dataset 6 Coverage Diagnostics](../../data/01_bronze/session_20260810_112047/shared_vision_labels_coverage_plot.png)

## Dataset 7: Aruco Markers + Hard coded camera + Multi-coloured Shapes

**Location:** `host_software\data\01_bronze\session_20260810_114330`
**Model Tag:** ``

This dataset is taken from the **same hard-coded angle** as Dataset 2 and 3, but the platform features a different piece of paper with Aruco markers. We used Aruco marker tracking to precisely determine the homography and location of the platform.

Part of new set that has the same aruco features. Multi-coloured markers are included
see aruco_markers_03.tex file for reference

### Processing log

```text
Loading .\host_software\data\01_bronze\session_20260810_114330\synced_telemetry.csv...
Initial rows: 26759
Removed 0 duplicate image rows.
Removed 3395 frozen telemetry frames (ball off-board or bouncing).
Final Cleaned rows: 23364
Saved cleaned dataset to .\host_software\data\01_bronze\session_20260810_114330\labels_sequential.csv

Reading dataset: host_software\data\01_bronze\session_20260810_114330\labels_sequential.csv
Total rows: 23364
Grid cell frequencies - min: 1, max: 451, median: 18
Frequency outlier threshold (Q3 + 1.5*IQR): 55.0
Target majority frequency: 17
Rows after normalization: 12888 (dropped 10476)
Saved normalized dataset to: host_software\data\01_bronze\session_20260810_114330\labels_normalized.csv
```

### Coverage Diagnostics

- **Coverage Score**: 99.48% (191/192 cells met the goal of 10 samples per 10x10mm)

![Dataset 7 Coverage Diagnostics](../../data/01_bronze/session_20260810_114330/shared_vision_labels_coverage_plot.png)

## Dataset 8: Combined dataset for multi-head CNN

**Location:** `host_software\data\03_gold\shared_vision`
**Model Tag:** ``

This dataset is a combination of the four previous datasets.

### Processing log

```text
Merged dataset written to host_software\data\03_gold\shared_vision
Per-session row counts:
  session_20260810_104132: 14834 merged (source had 14834)
  session_20260810_110239: 10988 merged (source had 10988)
  session_20260810_112047: 14067 merged (source had 14067)
  session_20260810_114330: 12883 merged (source had 12883)
OK: 52772 rows across 4 sessions; all images/masks present; no filename collisions.
```

### Coverage Diagnostics

- **Coverage Score**: 100.00% (192/192 cells met the goal of 10 samples per 10x10mm)

```text
--------------------------------------------------
COVERAGE METRICS:
Total safe zone area evaluated: -80 to 80 (X), -60 to 60 (Y)
Grid size: 2.5x2.5 mm (3072 total grid cells)
Goal: Minimum 1 samples per grid cell
Result: 3070/3072 cells met the goal.
Coverage Score: 99.93%
--------------------------------------------------
CELL DENSITY DISTRIBUTION (samples per grid cell):
  Empty cells (0 samples): 2/3072
  Min:    0
  P25:    12.0
  Median: 15.0
  Mean:   15.4
  P75:    19.0
  Max:    37
  Std:    5.7
--------------------------------------------------
COVERAGE AT OTHER THRESHOLDS:
  >=   0 samples/cell: 3072/3072 cells (100.00%)
  >=   9 samples/cell: 2717/3072 cells (88.44%)
  >=  18 samples/cell: 1109/3072 cells (36.10%)
  >=  28 samples/cell: 53/3072 cells (1.73%)
  >=  37 samples/cell: 2/3072 cells (0.07%)
--------------------------------------------------
```

![Combined Dataset Coverage Diagnostics](../../data/03_gold/shared_vision/labels_coverage_plot.png)
