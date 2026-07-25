# Source Libraries

The `src` directory contains the foundational, shared modules used by the primary application (`main.py`) to orchestrate the robot.

## Core Modules

- **`models.py`**: Model loading utilities for both PyTorch (`.pt`, `.pth`) and OpenVINO (`.xml`) models.
- **`receivers.py`**: High-performance classes for ingesting video data. Includes `UDPReceiver` for reading the Gigabit Ethernet stream from the FPGA, and `USBReceiver` for standard webcams.
- **`mock_receiver.py`**: A dummy receiver for testing the pipeline without active hardware.
- **`openvino_dispatcher.py`**: An asynchronous dispatcher wrapping `ov.AsyncInferQueue` to maximize throughput for OpenVINO models.
- **`state_machine.py`**: The central state manager (`TargetStateMachine`) that interprets audio commands and translates them into actionable X/Y target coordinates for the ball.
- **`audio_utils.py` / `utils.py`**: Various helper functions for calculating STFTs, auto-detecting COM ports, etc.
