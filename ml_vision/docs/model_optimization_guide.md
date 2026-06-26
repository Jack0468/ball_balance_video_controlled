# Model Deployment and Optimization Guide

When transitioning from the development phase to real-time execution, latency becomes the primary bottleneck. The standard Ultralytics YOLOv8 weights are distributed as `.pt` (PyTorch) files. While PyTorch is excellent for training and iteration, it incurs significant Python-level overhead during inference.

If the vision pipeline is failing to meet the >30 FPS requirement for the hardware PID control loop, you must bypass the PyTorch overhead by exporting the model to a highly optimized execution format.

## 1. ONNX (Open Neural Network Exchange)
ONNX is an open-standard format. Exporting to ONNX strips away the PyTorch framework bloat and compiles the model into a static mathematical graph. It runs on the ONNX Runtime engine (written in C++).

- **Target Hardware**: Best for standard CPUs.
- **Expected Performance Gain**: Typically a 2x to 3x speedup compared to standard `.pt` inference.
- **Export Command**:
  ```bash
  yolo export model=ml_vision/models/weights/yolov8n.pt format=onnx
  ```
  *(This will generate a `yolov8n.onnx` file in the same directory).*

## 2. TensorRT (NVIDIA GPUs Only)
TensorRT is NVIDIA's proprietary, ultra-high-performance deep learning inference engine. When exporting to TensorRT, the compiler physically analyzes the exact architecture of your local GPU (e.g., RTX 3080, GTX 1660) and compiles the neural network to perfectly map to your GPU's tensor cores. It merges layers, optimizes memory bandwidth, and calibrates precision (FP16/INT8).

- **Target Hardware**: NVIDIA GPUs (requires CUDA Toolkit and cuDNN).
- **Expected Performance Gain**: Often a 5x to 10x speedup compared to PyTorch. This is the absolute maximum frames-per-second physically possible on your hardware.
- **Export Command**:
  ```bash
  yolo export model=ml_vision/models/weights/yolov8n.pt format=engine
  ```
  *(This will generate a `yolov8n.engine` file).*

## Integration
Once you have exported your new weights (`.onnx` or `.engine`), simply update the path in your initialization scripts. Ultralytics automatically detects the file extension and will use the correct backend C++ runtime under the hood:

```python
from ultralytics import YOLO

# Standard slow PyTorch implementation
# model = YOLO("models/weights/yolov8n.pt")

# Fast ONNX CPU execution
model = YOLO("models/weights/yolov8n.onnx")

# Ultra-fast TensorRT GPU execution
# model = YOLO("models/weights/yolov8n.engine")
```
