# Model Evaluation Strategy

For our real-time ball balancing application, we need to find the optimal balance of speed, accuracy, and ease of deployment for a microcontroller/microprocessor setup. We can install any necessary Python modules for testing.

While YOLO and MobileNet SSD look to be the best candidates at a glance, we will run quick tests using all the approaches listed below to determine the best fit empirically.

## Cost-Benefit Analysis of ML Frameworks / Models

| Framework / Model | Pros | Cons | Use Case Recommendation |
| :--- | :--- | :--- | :--- |
| **OpenCV (Classical CV)** | Very fast inference, low compute overhead, highly interpretable. No training data required for initial setup. | Less robust to varying lighting, shadows, and background changes. | Best as a baseline or fallback if lighting is strictly controlled. |
| **YOLO (via PyTorch/ONNX)** | Extremely fast for object detection, robust to lighting and background variations. High accuracy. | Requires labeled data. Training overhead. Might be overkill if the environment is static. | Strong candidate. Best if lighting/background varies significantly and high robustness is needed. |
| **MobileNet SSD (TensorFlow/TFLite)** | Optimized for mobile/edge devices, very low latency. Good accuracy. | Slightly less accurate than the newest YOLO versions. Requires labeled data. | Strong candidate. Great balance for running directly on resource-constrained microprocessors. |
| **OpenCV DNN Module** | Can run pre-trained YOLO/SSD models without installing bulky PyTorch/TF libraries. Excellent for deployment. | Tricky to train within OpenCV itself (usually train in PyTorch/TF, export to ONNX, run in OpenCV). | Best for the final deployment inference pipeline. |

## Execution Plan

1. **Robust Data Processing:** Start by building a robust data processing pipeline to sync the raw MP4 video files with the CSV positional labels based on timestamps.
2. **Benchmark Training & Testing:** Train lightweight models (YOLOv8n and MobileNet SSD) and set up a classical OpenCV baseline. Evaluate them all on our synced dataset.
3. **Deployment:** Once the best model is identified through quick tests, deploy it using the OpenCV DNN module or ONNX Runtime to minimize dependencies and maximize real-time performance on the microprocessor.
