# Ball Balancing Robot: System Architecture

> [!NOTE]
> This document defines the macro-architecture for the Camera & Audio-Controlled Ball Balancing Robot using an Opal Kelly FPGA.

## 1. Hardware Topology

The system is a hybrid software/hardware architecture consisting of a Host PC running Machine Learning pipelines and an Opal Kelly FPGA driving the physical actuators.

- **Host PC**: Captures webcam video and microphone audio. Runs Python-based ML models for inference.
- **USB/PCIe Bridge**: Uses the Opal Kelly FrontPanel SDK (Python wrapper) to pass data seamlessly to the FPGA.
- **Opal Kelly FPGA**: Runs hardware-synthesized Verilog (converted via Vitis HLS) to perform Inverse Kinematics (IK) and fast PID loop calculations in pure silicon.
- **Actuators**: Three Nema 17 stepper motors (17HS3401S) operating the 3-DOF parallel manipulator platform.
- **Motor Drivers**: TMC2208 stepper motor drivers (rated at 2A) providing power to the Nema 17s. These drivers use a UART operational control interface, allowing the FPGA to dynamically configure microstepping, stealthChop parameters, and drive current over a serial line alongside the standard movement pulses.

## 2. Vision Subsystem (Python)

The vision system is responsible for continuous spatial tracking.
- **Inputs**: Real-time video feed from an overhead camera.
- **Preprocessing**: Uses Canny edge detection and contour approximation to dynamically find the platform boundary, then applies a perspective transformation to normalize the view.
- **Tracking Algorithm**: Uses a trained ML model (e.g., YOLOv8 or MobileNet SSD) to detect and bound the ball.
- **Coordinate Transformation**: The Python script is strictly responsible for transforming raw camera pixels into real-world physical coordinates (millimeters from the center) before transmission, offloading floating-point math from the FPGA.

## 3. Audio Subsystem (Python)

*Developed by an independent module team.*
- **Inputs**: Spoken user commands collected from multiple speakers.
- **Classification Model**: Uses an ML audio classifier to decode intent.
- **Command Set**: "go red", "go blue", "go green", "go yellow", "hold", and "stop".
- **Output**: Transmits a target state/color setpoint to the main execution loop.

## 4. Control Subsystem (FPGA Hardware)

To achieve maximum stability and latency reduction, the mechanical logic is decoupled from the Host PC's operating system.
- **High-Level Synthesis (HLS)**: The legacy Teensy C++ IK and PID algorithms are compiled directly into a Verilog RTL IP core using AMD-Xilinx Vitis HLS.
- **Interpolation**: The FPGA internally extrapolates the ball's position at a high frequency (e.g., 1000Hz) to maintain a fast PID feedback loop between the slower 60Hz camera frame updates.
- **Execution**: The FPGA receives the current physical `(x, y)` (in mm) from the Vision Model and the target state from the Audio Model via FrontPanel endpoints. The hardware IP core computes the exact step/direction pulses required and generates them continuously to drive the stepper motors.
