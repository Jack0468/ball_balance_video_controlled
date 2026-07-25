# Data Collection

This directory contains utility scripts to collect, sync, and log training data directly from the robot.

Because the system relies on high-speed hardware, these scripts interface heavily with the serial ports and the UDP video stream to scrape telemetry.

## Scripts

- **`collect_training_data.py`**: Runs a continuous loop to stream video frames and telemetry, packaging them together for training datasets.
- **`iphone_data_logger.py`**: Experimental script for fusing iOS sensor data (IMU/LiDAR) as an alternative source of truth.
- **`fpga_data_logging_plan.md`**: Design document detailing how we plan to increase data harvesting throughput directly from the Zynq FPGA.
