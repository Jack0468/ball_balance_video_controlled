# Realtime Visual Pipeline Features

This document outlines the core features and components of our realtime ML vision pipeline, which handles live webcam capture, platform detection, and ball tracking.

## Overview
The pipeline has evolved into a robust **Deep Learning Architecture** utilizing state-of-the-art models for both platform detection and ball tracking, fully replacing the legacy classical CV approaches.

1. **Stage 1 (Platform Pose Detection)**: A `YOLOv8-Pose` model is used to detect the platform. Rather than just returning a bounding box, it natively predicts the **4 exact corners** of the platform as keypoints.
2. **Stage 2 (Ball Tracking)**: A `ResNet18` expert tracker is used to directly regress the (x, y) physical coordinates of the ball from the cropped and warped platform image.

---

## Validated End-to-End Pipelines
Through rigorous inference benchmarks, we have tested and validated the following pipeline combinations (stages of execution stacked in sequence). The models can be mixed and matched depending on computational constraints:

1. **ResNet Standalone**: Direct regression using just `resnet` (assumes fixed camera angle/crop).
2. **YOLO Standalone**: Detection using just `yolo` (if it was trained to detect both platform and ball).
3. **YOLO + MLP**: `yolo` detection ➔ `mlp` temporal smoothing.
4. **Classical Aruco + CNN**: OpenCV `aruco` marker homography ➔ `cnn_2d_tracker`.
5. **Classical Aruco + CNN + MLP**: OpenCV `aruco` ➔ `cnn_2d_tracker` ➔ `mlp` temporal smoothing.
6. **YOLO + CNN**: `yolov8_platform_pose` ➔ `cnn_2d_tracker`.
7. **YOLO + CNN + MLP**: `yolov8_platform_pose` ➔ `cnn_2d_tracker` ➔ `mlp`.

---

## 1. Platform Extraction (YOLO-Pose)

The pipeline previously relied on a YOLO bounding box combined with classical HSV color masking and morphological operations to find the platform corners. 
**Current Approach:** We now use a custom-trained **YOLOv8-Pose** model (`yolov8_platform_pose_markers`).
- **Input**: Full camera frame (e.g., 640x480).
- **Output**: Bounding box + 4 Keypoints representing the top-left, top-right, bottom-right, and bottom-left corners of the platform.
- **Advantage**: It is completely impervious to lighting changes, textured backgrounds, and color variations that would break classical CV HSV masking.

The 4 predicted keypoints are then used to compute a Homography matrix (`cv2.getPerspectiveTransform`), allowing us to warp the platform into a perfectly flat, top-down 2D plane.

---

## 2. Ball Tracking Models

Once the platform is warped into a top-down view, it is passed to our dedicated tracker.

### ResNet18 Expert Tracker (`resnet18_expert_tracker`)
- **Architecture**: A modified ResNet-18 model with its final fully connected layer replaced to output 2 continuous values (X and Y coordinates).
- **Performance**: Achieves exceptional sub-pixel accuracy (~5.9mm Mean Euclidean Error).
- **Input**: Warped 240x320 RGB image.
- **Output**: (x, y) coordinates mapped to the physical `200.0mm` bounds.

### Legacy Models
- **CNN 2D Tracker**: A lighter, custom `BasicCNN` architecture.
- **Classical CV Model**: An HSV color-masking and Hough Circle model. Relies on strict thresholding and is highly susceptible to lighting changes.
- **Target Marker Tracker**: A shape-first, color-second classical CV algorithm that finds circular markers.

---

## 3. Temporal Correction (MLP Corrector)

To further stabilize the coordinates and eliminate any high-frequency jitter, the raw (X, Y) predictions from the vision models can be passed through a lightweight Multi-Layer Perceptron (MLP) corrector.
- The MLP takes the vision coordinate, the current target coordinate, and the time delta `dt`.
- It acts as an intelligent, learned Kalman filter to smooth the trajectory for the control systems.

---

## 4. References & Theory

The mathematical principles underpinning this pipeline's projective geometry—specifically the mapping of 3D real-world coordinates into a 2D plane via Homogeneous Coordinates, Pinhole Camera Models, and Homography (`cv2.getPerspectiveTransform`)—are deeply grounded in the core literature of computer vision. 

For a comprehensive breakdown of the underlying math driving this physics-based tracking, please refer to the core textbook:
- [Foundations of Computer Vision: 2D Motion from 3D](https://github.com/Foundations-of-Computer-Vision/visionbook/blob/main/2d_motion_from_3d.qmd)
