import cv2
import numpy as np
import os
import openvino as ov

script_dir = os.path.dirname(os.path.abspath(__file__))
yolo_xml = os.path.abspath(os.path.join(script_dir, 'ml_vision/models/yolo_platform_markers_v2/weights/best_openvino_model/best.xml'))

core = ov.Core()
yolo_model = core.read_model(yolo_xml)
yolo_compiled = core.compile_model(yolo_model, "CPU")

# Dummy frame (640x640)
padded_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

# Preprocess
img = cv2.cvtColor(padded_frame, cv2.COLOR_BGR2RGB)
img = img.transpose((2, 0, 1))[np.newaxis, ...].astype(np.float32) / 255.0

print(f"Input shape: {img.shape}")
print(f"Input memory is C-contiguous: {img.flags['C_CONTIGUOUS']}")

# Run inference
res = yolo_compiled([img])
output = list(res.values())[0]

print(f"Output shape: {output.shape}")

# Simulate parsing
yolo_res = output.copy()
boxes_transposed = yolo_res[0].T  # (8400, 29)
print(f"Boxes transposed shape: {boxes_transposed.shape}")

num_classes = 13
class_scores = np.max(boxes_transposed[:, 4:4+num_classes], axis=1)
class_ids = np.argmax(boxes_transposed[:, 4:4+num_classes], axis=1)

mask = class_scores > 0.5
filtered_boxes = boxes_transposed[mask]

print(f"Found {len(filtered_boxes)} boxes with score > 0.5")
print(f"Max class score across all 8400 boxes: {np.max(class_scores):.4f}")
print("Test completed.")
