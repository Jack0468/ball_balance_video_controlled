# Host Software

The `host_software` directory serves as the "Brain" of the VRI 2026 ball-balancing robot. It is entirely written in Python and runs on a Host PC (Laptop). It performs heavy ML inferences using the laptop's internal microphone and an attached USB webcam, then communicates via USB Serial with the STM32 microcontroller, which handles edge-level control.

## Architecture

The system uses a highly modular architecture relying heavily on PyTorch for Machine Learning tasks:
- **Vision (Laptop)**: YOLOv8 pose estimation determines the 3D orientation of the platform and the position of the ball in real-time from a USB camera feed.
- **Audio (Laptop)**: A threaded audio classifier listens to the internal microphone for vocal commands to change the robot's target state.
- **Control (STM32)**: The host PC sends vision state and target commands to the STM32 over USB. The STM32 natively hosts the weights of the `ml_control` policy (Reinforcement Learning) alongside classical controllers (PID/IK) to output precision motor steps.

## Primary Entry Point

The main operational script is `main.py`. This script integrates the camera receiver, the PyTorch YOLO model, the PyTorch Audio Classifier, and the State Machine to fully operate the robot.

```bash
python main.py --udp
```

## Directory Structure

- `/ml_vision`: Computer vision pipelines, dataset generation, YOLO training, and core homography mathematics.
- `/ml_audio`: PyTorch-based audio classification models and real-time audio receiver logic.
- `/ml_control`: Experimental ML control policies (e.g., Reinforcement Learning via STM32).
- `/src`: Shared utility libraries, model loaders, receivers, and state machine logic used by `main.py`.
- `/control`: Classical control algorithms including PID controllers and inverse kinematics.
- `/data_collection`: Telemetry and logging scripts for aggressive data acquisition from the FPGA.
- `/tools`: Utility scripts, debugging bridges, and an archive of legacy inference loops and alternative OpenVINO `main` scripts.
