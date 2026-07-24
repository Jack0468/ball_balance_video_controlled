import sys
from ultralytics import YOLO
import cv2

yolo_ov = YOLO('ml_vision/models/yolo_platform_markers_v2/weights/best_openvino_model/', task='pose')
yolo_pt = YOLO('ml_vision/models/platform_and_markers_model/weights/best_openvino_model/', task='pose')

img = cv2.imread('ml_vision/data/yolo_raw_dataset/images/0000.jpg')

res_ov = yolo_ov.predict(source=img, imgsz=640, conf=0.5, verbose=False)[0]
res_pt = yolo_pt.predict(source=img, imgsz=640, conf=0.5, verbose=False)[0]

print('V2 OpenVINO boxes:', res_ov.boxes.cls.cpu().numpy() if res_ov.boxes is not None else None)
print('PT OpenVINO boxes:', res_pt.boxes.cls.cpu().numpy() if res_pt.boxes is not None else None)
