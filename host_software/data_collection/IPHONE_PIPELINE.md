# Legacy iPhone Data Pipeline

This document outlines the legacy architecture used to collect the 5 original iPhone videos for the expert PID tracker.

> [!WARNING]
> This dataset was collected using a rapid sweeping state machine that was not semantic. **It cannot be used for VLA language training.**

## 1. Data Collection (`iphone_data_logger.py`)
- The iPhone recorded a 4K 60fps or 1080p 60fps video.
- Simultaneously, `iphone_data_logger.py` read the binary `TelemetryPacket` directly from the STM32 via USB and wrote `iphone_telemetry.csv`.
- The script flashed a dark gray/black background every 500ms to allow visual synchronization with the iPhone camera.

## 2. Synchronization (`sync_data.py`)
- The iPhone `.MOV` or `.MP4` files were copied to `01_bronze/videoX/`.
- `sync_data.py` was used to precisely align the blinking screen in the video with the timestamps in `iphone_telemetry.csv`.
- The script extracted the aligned frames into `02_silver/images/` and produced a unified `labels_sequential.csv`.

## 3. Usage
This dataset was used exclusively to train the pure Vision tracker (`ml_vision` YOLO and MLP) to emulate the PID expert, as well as to validate the basic camera-to-motor latency.
