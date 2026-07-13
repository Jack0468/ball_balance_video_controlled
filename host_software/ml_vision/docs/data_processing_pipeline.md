# Data Processing Pipeline

This document outlines the exact offline data processing pipeline used to convert raw iOS video recordings and raw Python telemetry logs into synchronized, cropped, and aligned 640x480 images mapped directly to physical coordinates for ML training.

## The Problem
1. **Variable Frame Rate (VFR)**: iOS devices do not record video at a perfect 30.0 FPS. They record at a variable frame rate, meaning we cannot assume `Frame 150` happened exactly at `5.000` seconds.
2. **Clock Drift**: The iPhone camera and the Python Telemetry logger run on completely separate devices. The Python logger could be started minutes before the iPhone begins recording.

## The Solution: Frame-Aware Anchoring
To solve these issues, we rely on a **Visual Timestamp**. 
The Python telemetry logger draws a giant green Unix Timestamp on the laptop screen in real-time. By reading this timestamp visually off the screen in the video, we can mathematically perfectly anchor one specific frame of the video to one specific row of the telemetry.

From that anchor frame, OpenCV can extract the exact presentation timestamp (`pos_msec`) of every other frame in the video to map it to the telemetry data perfectly, fully negating Variable Frame Rate issues.

---

## The Workflow

### Step 1: Read the True Timestamp Visually
Run the interactive timestamp finder script. Use your `Right/Left Arrow` keys (or `e`/`q` to jump) to step forward into the video until the laptop screen is visible and you clearly see the giant green number.
Write down BOTH the **Frame Index** (shown in red at the top left) and the **Giant Green Timestamp** (shown on the laptop).
```powershell
conda run -n ball_balance_env python host_software/ml_vision/data_processing/get_timestamp.py --video host_software/ml_vision/data/01_bronze/video1/20260710_054604000_iOS.MOV
```

### Step 2: Synchronize Data
Feed BOTH the Frame Index and the Green Timestamp into the synchronizer. This script analyzes the Variable Frame Rate offsets, takes ~10 seconds, and creates a `synced_telemetry.csv` file perfectly mapping every video frame to the correct physics row.
```powershell
conda run -n ball_balance_env python host_software/ml_vision/data_processing/sync_data.py --video host_software/ml_vision/data/01_bronze/video1/20260710_054604000_iOS.MOV --telemetry ml_vision/data/bronze/iphone_telemetry.csv --sync-frame <FRAME_INDEX> --sync-timestamp <YOUR_GREEN_NUMBER> --output host_software/ml_vision/data/01_bronze/video1/synced_telemetry.csv
```

### Step 3: Fast Sync Verification (Optional but Recommended)
Before running the 30-minute full preprocessing script, run the checker script to instantly generate a 10-second verification video from the raw `.MOV` file. It will overlay the physical target positions onto the video based on your synchronized CSV. If the red circle tracks perfectly with the finger in the video, the sync is flawless.
```powershell
conda run -n ball_balance_env python host_software/ml_vision/data_processing/check_sync.py --video host_software/ml_vision/data/01_bronze/video1/20260710_054604000_iOS.MOV --synced-csv host_software/ml_vision/data/01_bronze/video1/synced_telemetry.csv --crop "82,435,915,762" --start-idx 3000
```
*(You can change `--start-idx` to check any specific part of the video.)*

### Step 4: Preprocess Dataset
Run the computer vision preprocessor to apply your crop box, resize all frames to standard 640x480, and extract the final images. This will take ~20-30 minutes for a long video. The resulting output is stored as `02_silver` data ready for ML training.
```powershell
conda run -n ball_balance_env python host_software/ml_vision/data_processing/preprocess_dataset.py --video host_software/ml_vision/data/01_bronze/video1/20260710_054604000_iOS.MOV --synced-csv host_software/ml_vision/data/01_bronze/video1/synced_telemetry.csv --crop "82,435,915,762"
```

---

## Coordinate System Reference
The machine learning pipeline and the physical robot (`Screen.cpp`) operate on a standard Cartesian 2D plane:
- **Origin (0,0)**: The exact physical center of the platform (and the exact center of the 640x480 cropped image at `320, 240`).
- **Width**: `140mm` (-70mm to +70mm)
- **Height**: `110mm` (-55mm to +55mm)
- **X Axis**: Increases from Left (-70) to Right (+70)
- **Y Axis**: Increases from Bottom (-55) to Top (+55)
