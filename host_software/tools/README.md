# Tools & Legacy

This directory contains standalone utility scripts for hardware debugging, as well as an archive of legacy code.

## Subdirectories

- **`legacy_mains/`**: Contains older iterations of the main application entry point (e.g., `main_resnet.py`, `main_yolo_pytorch.py`, `main_yolo.py`). These represent previous milestones (like the ResNet expert tracker or OpenVINO async implementations) that are no longer actively used but kept for reference.
- **`legacy_inference/`**: A collection of prototype inference loops previously stored in the `ml_vision` root.
- **`AccelStepper/` / `Python Plotters/`**: Various hardware and diagnostic tools.

## Scripts

- **`receive_udp_video.py`**: A diagnostic script that connects to the Zynq FPGA's Gigabit Ethernet UDP stream, parses the packets, and renders the raw video feed to a window. Useful for verifying camera hardware before launching the full ML stack.
- **`test_motors.py`**: Script to send direct step commands to the stepper motors via serial.
- **`fpga_bridge.py` & `python_multiplexer.py`**: Diagnostic bridges for communicating with the FPGA directly.
