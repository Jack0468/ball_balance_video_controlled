# VLA in Machine Learning: Multimodal Expert Systems for Autonomous Edge Robotics

> [!WARNING]
> **ARCHIVED 2026-09-12.** Draft poster describing the vision expert as YOLOv8/ResNet and
> predates the Jetson AGX Orin / photonic 3-arm comparison framing. Current locked vision
> architecture is the Shared Backbone CNN (see `CLAUDE.md`); the project's current major
> goal is the Multimodal/VLA 3-arm comparison (see `host_software/ml_jetson_vla/docs/ARCHITECTURE.md`).
> Kept for historical reference only — do not cite these results as current.

## Introduction
Autonomous balancing robots operating in unstructured environments demand ultra-low latency inference and multimodal reasoning. Classical monolithic control networks often suffer from "black-box" unpredictability and sluggish inference times on edge hardware. This project proposes a **Distributed Expert Architecture**, decoupling Vision (YOLOv8/ResNet), Audio/Language (Wav2Vec), and Action (STM32 PID). By evaluating this expert-driven topology against a unified Vision-Language-Action (VLA) Behavioral Cloning model, we demonstrate that modular intelligence achieves superior determinism and sub-30ms end-to-end latency without sacrificing capability.

## Materials and Methods
Our 3-DOF parallel manipulator platform is driven by three Nema 17 stepper motors, dynamically balancing a free-rolling ball. The system employs a **Medallion Architecture** data pipeline (`Bronze -> Silver -> Gold`) to train specialized expert models:
- **Vision Expert**: YOLOv8 extracts platform bounds and ball position via homography, achieving sub-millimeter precision.
- **Audio Expert**: A lightweight CNN classifies microphone waveforms into discrete user commands ("go red", "hold").
- **Action Expert**: An STM32F407G microcontroller executes high-frequency (1000Hz) Inverse Kinematics and PID control loops.

For comparison, a unified VLA model was trained using Imitation Learning to directly predict motor targets (`theta_a`, `theta_b`, `theta_c`) from the temporally aligned multi-modal token stream.

## Results

![Model Comparisons (Accuracy vs Frames Per Second)](model_comparisons.png)

![System Latency Distribution](system_latency_distribution.png)

The decoupled Expert Architecture achieved an average end-to-end processing latency of **25.7ms**. The YOLOv8 pose models delivered superior spatial accuracy (RMSE < 2mm) operating at ~80 FPS on edge accelerators, outperforming the generalized VLA baseline in both reaction time and zero-shot robustness. 

## Conclusions
Decoupling robotic intelligence into highly specialized Vision, Audio, and Action experts provides critical advantages for edge deployments. While end-to-end VLA models offer elegant architectural simplicity, they currently trail behind distributed expert systems in low-latency physical control tasks. Future work will investigate hybrid architectures that leverage a VLA model for high-level semantic planning while retaining deterministic PID loops for low-level motor actuation.

## Acknowledgments
We would like to acknowledge the University of Sydney's Electrical Engineering department for providing the laboratory space and structural components necessary for the robotic platform.

## Further Information
- **Jack McCudden** - Electrical Engineering, The University of Sydney
- **Project Repository**: [GitHub Link]
- **Contact**: jmcc0000@uni.sydney.edu.au
