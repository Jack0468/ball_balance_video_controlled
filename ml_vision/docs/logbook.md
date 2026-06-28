# Implementation Logbook

## 2026-06-24
- **12:50** - Initiated execution phase for the ball position vision model.
- **12:51** - Created task list and this logbook document.
- **12:51** - Started implementing Data Processing Pipeline (`video_sync.py`, `camera_calibration.py`).
- **12:52** - Implemented `preprocessor.py` for ArUco and perspective transforms.
- **12:53** - Implemented Models Benchmarking suite (`classical_cv_model.py`, `ml_model_benchmarks.py`).
- **12:53** - Implemented Evaluation & Output (`evaluator.py`, `output_formatter.py`).
- **12:54** - Running syntax verification on all implemented python scripts.
- **12:54** - Syntax verification passed successfully.
- **12:55** - Completed execution of all tasks. Walkthrough generated.

## 2026-06-25
- **10:50** - Approved implementation plan for real-time webcam testing of the vision pipeline.
- **10:50** - Decision: We will support both GUI rendering and headless execution in the test scripts so we can prioritize testing model accuracy/reliability first, and then measure raw latency.
- **10:50** - Decision: For the Edge Detection fallback (if Canny fails to find the platform), we will implement both options (Skip Inference entirely vs. Run YOLO on unwarped raw frame) so we can back-test which option yields better results when we collect more data.

## 2026-06-26 - 2026-06-28
- **13:30 (Jun 26)** - Implemented first pass of `marker_tracker.py` to identify dynamic target destinations via HSV color filtering.
- **13:50 (Jun 26)** - **Algorithm Pivot**: Rewrote marker tracking to a "Shape-First, Color-Second" methodology (`cv2.adaptiveThreshold` + `cv2.findContours` Circularity filter) to fix critical tracking failures on low-brightness (Black/Grey) physical targets.
- **14:50 (Jun 26)** - Commenced major Medallion Architecture dataset overhaul.
- **22:50 (Jun 27)** - Concluded data module refactor. Created `01_bronze`, `02_silver`, and `03_gold` lakes. Migrated all HEIC, MP4, and synthetic dataset files.
- **22:52 (Jun 27)** - Updated `augment_dataset.py`, `convert_heic.py`, and all YOLO config YAML files to route properly through the new Medallion pipeline.
- **23:59 (Jun 27)** - Updated the root `.gitignore` to mask the new massive dataset directories from version control, saving the git repository from bloating.
