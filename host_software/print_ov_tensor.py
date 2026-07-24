import openvino as ov
import numpy as np
import cv2

core = ov.Core()
yolo_compiled = core.compile_model('ml_vision/models/yolo_platform_markers_v2/weights/best_openvino_model/best.xml', 'CPU')

img = cv2.imread('ml_vision/data/02_silver_unified_pose/images/train/frame_0000.jpg')
if img is None:
    print("Image not found. Creating random image.")
    img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
else:
    img = cv2.resize(img, (640, 640))

# OpenVINO with embedded reverse_channels and scale_values expects BGR 0-255
img = img.transpose((2, 0, 1))[np.newaxis, ...].astype(np.float32)
img = np.ascontiguousarray(img)

res = yolo_compiled([img])
output = list(res.values())[0]

boxes = output[0].T
print(f'Boxes shape: {boxes.shape}')

# Find the best box (assume class 0 is index 4)
best_idx = np.argmax(boxes[:, 4])
best_box = boxes[best_idx]

print(f'Best box score for class 0: {best_box[4]:.3f}')
print(f'Box coords (x,y,w,h): {best_box[0:4]}')
print(f'Classes (4 to 16): {best_box[4:17]}')
print(f'Keypoints (17 to 28): {best_box[17:29]}')
