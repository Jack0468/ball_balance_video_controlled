import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# Physical Marker Coordinates in Platform-Centric Millimeters
# Origin is Top-Left of the platform
PLATFORM_W = 187.5
PLATFORM_H = 142.0

MARKERS_PHYSICAL_MM = [
    (33.0, 26.0),                   # Green (Top-Left)
    (187.5 - 41.0, 53.0),           # Red (Top-Right)
    (187.5 - 13.0, 142.0 - 8.0),    # Black (Bottom-Right)
    (69.0, 142.0 - 58.0)            # Grey (Bottom-Left)
]

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.abspath(os.path.join(script_dir, "data/02_silver/images"))
    telemetry_path = os.path.abspath(os.path.join(script_dir, "data/02_silver/labels.csv"))
    
    output_dir = os.path.abspath(os.path.join(script_dir, "data/04_auto_labeled"))
    out_images_dir = os.path.join(output_dir, "images")
    out_labels_dir = os.path.join(output_dir, "labels")
    
    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_labels_dir, exist_ok=True)
    
    model_path = os.path.abspath(os.path.join(script_dir, "models/yolov8_marker_and_ball_detector/weights/best.pt"))
    if not os.path.exists(model_path):
        print(f"Error: Could not find YOLO model at {model_path}")
        return
        
    print("Loading YOLO model for marker detection...")
    model = YOLO(model_path)
    
    print("Loading telemetry...")
    df = pd.read_csv(telemetry_path)
    # create a dict for fast lookup: image_file -> row
    telemetry_dict = {}
    for _, row in df.iterrows():
        telemetry_dict[str(row['image_file'])] = row
        
    image_files = [f for f in os.listdir(input_dir) if f.endswith('.jpg') or f.endswith('.png')]
    # Take a subset of 2000 for quick training
    image_files = image_files[:2000]
    
    print(f"Generating labels for {len(image_files)} images...")
    
    success_count = 0
    for idx, img_name in enumerate(image_files):
        img_path = os.path.join(input_dir, img_name)
        
        # We need telemetry for this frame
        if img_name not in telemetry_dict:
            continue
            
        row = telemetry_dict[img_name]
        touch_x = row['touch_x']
        touch_y = row['touch_y']
        
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        h, w = img.shape[:2]
        
        # Run YOLO to find markers
        results = model.predict(source=img, imgsz=640, conf=0.5, verbose=False)
        result = results[0]
        
        markers = {}
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            if cls_id >= 1 and cls_id <= 4:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                bw = x2 - x1
                bh = y2 - y1
                markers[cls_id] = (cx, cy, bw, bh)
                
        # We need exactly all 4 markers to compute homography reliably (Green=1, Red=2, Black=3, Grey=4)
        if len(markers) == 4 and all(i in markers for i in [1, 2, 3, 4]):
            # Source pixels
            src_pixels = np.array([
                [markers[1][0], markers[1][1]],
                [markers[2][0], markers[2][1]],
                [markers[3][0], markers[3][1]],
                [markers[4][0], markers[4][1]]
            ], dtype=np.float32)
            
            dst_mm = np.array(MARKERS_PHYSICAL_MM, dtype=np.float32)
            
            # M maps Pixels -> MM, M_inv maps MM -> Pixels
            M, _ = cv2.findHomography(src_pixels, dst_mm)
            if M is None:
                continue
            try:
                M_inv = np.linalg.inv(M)
            except np.linalg.LinAlgError:
                continue
                
            # Project ball MM to Pixels
            ball_x_mm = touch_x + (PLATFORM_W / 2.0)
            ball_y_mm = touch_y + (PLATFORM_H / 2.0)
            
            ball_pt_mm = np.array([[[ball_x_mm, ball_y_mm]]], dtype=np.float32)
            ball_pt_pixel = cv2.perspectiveTransform(ball_pt_mm, M_inv)[0][0]
            bx, by = ball_pt_pixel[0], ball_pt_pixel[1]
            
            # YOLO Format: class_id center_x center_y width height (normalized 0-1)
            labels = []
            
            # Class 0: Ball (assume 30x30 pixels)
            bx_norm = max(0, min(1, bx / w))
            by_norm = max(0, min(1, by / h))
            bw_norm = 30.0 / w
            bh_norm = 30.0 / h
            labels.append(f"0 {bx_norm:.6f} {by_norm:.6f} {bw_norm:.6f} {bh_norm:.6f}")
            
            # Markers 1-4
            for cls_id, (mcx, mcy, mbw, mbh) in markers.items():
                labels.append(f"{cls_id} {mcx/w:.6f} {mcy/h:.6f} {mbw/w:.6f} {mbh/h:.6f}")
                
            # Save
            out_img_path = os.path.join(out_images_dir, img_name)
            out_txt_path = os.path.join(out_labels_dir, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))
            
            # We can just copy the image or write it
            cv2.imwrite(out_img_path, img)
            with open(out_txt_path, 'w') as f:
                f.write('\n'.join(labels) + '\n')
                
            success_count += 1
            if success_count % 100 == 0:
                print(f"Generated {success_count} labeled images...")
                
    print(f"Done! Successfully generated {success_count} auto-labeled images.")
    
    # create dataset.yaml
    yaml_content = f"""path: /content/ball_balance_video_controlled/host_software/ml_vision/data/04_auto_labeled
train: images
val: images

names:
  0: ball
  1: green_marker
  2: red_marker
  3: black_marker
  4: grey_marker
"""
    with open(os.path.join(output_dir, "dataset.yaml"), "w") as f:
        f.write(yaml_content)
    print("Created dataset.yaml for YOLO.")

if __name__ == '__main__':
    main()
