# VRI 2026 AI Agent Rules

## ML Vision Structure Requirements

All Machine Learning additions to this repository MUST strictly adhere to the established logical sub-directory structure within `host_software/ml_vision/`. 

Never dump Python scripts into a generic `scripts/` directory.

### Allowed Directories:
- `core/`: Foundational library modules (`preprocessor.py`, `coordinate_math.py`, `camera_calibration.py`, etc.)
- `data_processing/`: Video and telemetry processing tools used to generate `02_silver` data.
- `training/`: Model generation, YOLO pose architecture, and ONNX/OpenVINO export scripts.
- `tests/`: Functional validation and unit tests (e.g. `test_auto_crop.py`, `realtime_pipeline_test.py`). Not for ML metrics!
- `evaluations/`: Metric generation, model benchmarking, and plots (e.g. `evaluate_expert_tracker.py`).

If you are creating a new script, determine its purpose and place it in the most appropriate folder above.
