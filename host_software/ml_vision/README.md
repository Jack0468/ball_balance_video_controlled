# ML Vision

This directory contains the computer vision pipeline for the Ball-Balancing Robot. It focuses heavily on training and evaluating YOLO models for 3D pose estimation of the platform and object detection for the ball.

## Directory Structure

In adherence to project rules, all machine learning additions follow this logical sub-directory structure:

- **`core/`**: Foundational library modules for the vision pipeline. Contains algorithms for camera calibration, perspective transformation (`coordinate_math.py`), classical computer vision approaches, and preprocessors.
- **`data_processing/`**: Scripts and tools to process raw telemetry and video into unified, ML-ready datasets (Silver tier).
- **`training/`**: Model generation, YOLO pose architectures, and scripts for ONNX/OpenVINO exports.
- **`tests/`**: Functional validation and unit tests for the pipeline (e.g., `realtime_pipeline_test.py`, `test_auto_crop.py`). Note: ML metrics are not calculated here.
- **`evaluations/`**: Scripts for ML metric generation, model benchmarking, plotting, and analysis of model accuracy.
