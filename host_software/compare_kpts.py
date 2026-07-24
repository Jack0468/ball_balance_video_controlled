import openvino as ov
import numpy as np
import cv2
import torch
from ultralytics import YOLO

# Load PyTorch model
pt_model = YOLO('ml_vision/models/yolo_platform_markers_v2/weights/best.pt')

# Load OpenVINO model
core = ov.Core()
ov_model = core.compile_model('ml_vision/models/yolo_platform_markers_v2/weights/best_openvino_model/best.xml', 'CPU')

# Load image
img = cv2.imread('ml_vision/data/yolo_raw_dataset/images/0000.jpg')
if img is None:
    print("Cannot find image!")
    exit(1)
img = cv2.resize(img, (640, 640))

# 1. Run PyTorch (handles color/scaling internally)
print("--- PyTorch ---")
res = pt_model(img, verbose=False)[0]
pt_kpts = res.keypoints.xy[0].cpu().numpy()
print(f"PyTorch Keypoints:\n{pt_kpts}")

# 2. Run OpenVINO (requires manual BGR 0-255 because of embedded xml nodes)
print("\n--- OpenVINO ---")
ov_img = img.transpose((2, 0, 1))[np.newaxis, ...].astype(np.float32)
ov_img = np.ascontiguousarray(ov_img)

res_ov = ov_model([ov_img])
ov_output = list(res_ov.values())[0][0].T  # (8400, 29)

class_scores = np.max(ov_output[:, 4:17], axis=1)
best_idx = np.argmax(class_scores)
best_box = ov_output[best_idx]

print(f"Best score: {class_scores[best_idx]:.3f}")
kpts_raw = best_box[17:29]
ov_kpts = np.array([
    [kpts_raw[0], kpts_raw[1]],
    [kpts_raw[3], kpts_raw[4]],
    [kpts_raw[6], kpts_raw[7]],
    [kpts_raw[9], kpts_raw[10]]
])
print(f"OpenVINO Keypoints:\n{ov_kpts}")
