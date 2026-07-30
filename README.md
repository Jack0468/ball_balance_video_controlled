# VRI 2026: Ball-Balancing Robot

### Bronze System Demo

Demonstration of the integrated system balancing the ball using early iterations of the vision pipeline.
![Bronze System Demo](host_software/data/bronze_demo.mp4)

### Raw RGB Camera Feed

Raw 60fps camera feed captured for the ML datasets before homography warping is applied.
![Raw RGB Camera Feed](docs/assets/rgb_video_demo.gif)

### Vision/Platform Sync Check

Diagnostic visualization validating the latency and synchronization between camera frames and the physical platform kinematics.
![Vision/Platform Sync Check](docs/assets/sync_check_demo.gif)

A fully autonomous, multi-modal ball-balancing robot built using a highly distributed architecture. This system leverages Machine Learning (Computer Vision and Audio Classification), high-speed FPGA hardware acceleration, and precision stepper motor control to dynamically balance a ball on a moving platform.

## 🏗️ System Architecture

This monorepo is divided into three primary domain pillars:

### 1. Host Software (`/host_software`)

The "Brain" of the robot, written entirely in Python and running on the Host PC.

- **Machine Learning Vision (`/ml_vision`)**: Utilizes YOLOv8 for real-time tracking of the platform's 3D pose and orientation based on the camera feed.
- **Machine Learning Audio (`/ml_audio`)**: A voice command classifier that allows users to issue verbal instructions to the robot (e.g., "Balance", "Stop", "Move Left").
- **Data Collection (`/data_collection`)**: Scripts designed to aggressively pull raw video frames and touchscreen coordinate data from the FPGA over USB to build robust training datasets.

### 2. Edge Control Firmware (`/firmware`)

The active controller running on an **STM32 microcontroller**.
It runs the high-speed motor control loops and actively hosts the exported weights of the `ml_control` policy (e.g. Reinforcement Learning models) to perform edge inference for stabilization. It receives high-level states (Vision and Audio commands) from the Host PC via USB Serial.

### 3. FPGA Hardware Research (`/fpga`)

Experimental/Alternative architecture using a **Zynq-7000 (ZedBoard)** (and historically Opal Kelly XEM3010).
Designed to act as a deterministic, high-bandwidth bridge for streaming gigabit UDP video and high-frequency touch ADC data directly to the host.

### 4. Physical Hardware (`/hardware` & `/docs`)

Contains all the mechanical and electrical blueprints.

- **`/hardware/cad`**: 3D printable STL files and CAD assemblies for the robot's mechanical structure.
- **`/docs`**: Circuit diagrams, wiring schematics, and rigorous inverse kinematics derivations.

## 🚀 Getting Started

### 1. STM32 Firmware

1. Open the `/firmware/stm32_ml_control_and_vision` project in PlatformIO or Arduino IDE.
2. Compile and flash the code to your STM32.
3. Ensure the STM32 is connected to the Host PC via USB.

### 2. Python Environment

To install the necessary host software dependencies:

```bash
conda env create -f environment.yml
# OR
pip install -r requirements.txt
```

### 3. Running the Robot

1. Ensure the USB Webcam and the STM32 are plugged into the Host PC.
2. Execute the primary host software pipeline (handles Vision, Audio, and communication with the STM32):

```bash
cd host_software
python main.py
```
