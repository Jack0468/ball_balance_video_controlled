import os
import cv2
import numpy as np
import pandas as pd
import random
from ultralytics import YOLO

# Physical dimensions of the purple board (in mm)
PLATFORM_W = 187.5
PLATFORM_H = 142.0

# The physical corners of the board (Top-Left origin)
BOARD_CORNERS_MM = np.array([
    [0.0, 0.0],                     # Top-Left
    [PLATFORM_W, 0.0],              # Top-Right
    [PLATFORM_W, PLATFORM_H],       # Bottom-Right
    [0.0, PLATFORM_H]               # Bottom-Left
], dtype=np.float32)

# Physical coordinates of the 4 target markers (Top-Left origin)
TARGETS_PHYSICAL_MM = np.array([
    [33.0, 26.0],                   # Green (Top-Left)
    [187.5 - 41.0, 53.0],           # Red (Top-Right)
    [187.5 - 13.0, 142.0 - 8.0],    # Black (Bottom-Right)
    [69.0, 142.0 - 58.0]            # Grey (Bottom-Left)
], dtype=np.float32)

def order_corners(pts):
    # Sorts 4 points into: Top-Left, Top-Right, Bottom-Right, Bottom-Left
    pts = sorted(pts, key=lambda x: x[1]) # Sort by Y
    top = pts[:2]
    bottom = pts[2:]
    
    tl = min(top, key=lambda x: x[0])
    tr = max(top, key=lambda x: x[0])
    bl = min(bottom, key=lambda x: x[0])
    br = max(bottom, key=lambda x: x[0])
    
    return np.array([tl, tr, br, bl], dtype=np.float32)

def generate_random_warp(img_w, img_h):
    pts1 = np.float32([[0,0], [img_w,0], [img_w,img_h], [0,img_h]])
    jitter_x = int(img_w * 0.15)
    jitter_y = int(img_h * 0.15)
    
    pts2 = np.float32([
        [random.randint(-jitter_x, jitter_x), random.randint(-jitter_y, jitter_y)],
        [img_w + random.randint(-jitter_x, jitter_x), random.randint(-jitter_y, jitter_y)],
        [img_w + random.randint(-jitter_x, jitter_x), img_h + random.randint(-jitter_x, jitter_x)],
        [random.randint(-jitter_x, jitter_x), img_h + random.randint(-jitter_x, jitter_x)]
    ])
    
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    return matrix

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.abspath(os.path.join(script_dir, "data/02_silver/images"))
    telemetry_path = os.path.abspath(os.path.join(script_dir, "data/02_silver/labels.csv"))
    
    output_dir = os.path.abspath(os.path.join(script_dir, "data/06_dynamic_warped_dataset"))
    out_images_dir = os.path.join(output_dir, "images")
    out_labels_dir = os.path.join(output_dir, "labels")
    
    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_labels_dir, exist_ok=True)
    
    pose_model_path = os.path.abspath(os.path.join(script_dir, "models/platform_pose_model/weights/best.pt"))
    if not os.path.exists(pose_model_path):
        print(f"Error: Could not find pose model at {pose_model_path}")
        return
        
    print("Loading YOLO-Pose model...")
    pose_model = YOLO(pose_model_path)
    
    print("Loading telemetry...")
    df = pd.read_csv(telemetry_path)
    telemetry_dict = {str(row['image_file']): row for _, row in df.iterrows()}
        
    image_files = [f for f in os.listdir(input_dir) if f.endswith('.jpg') or f.endswith('.png')]
    image_files = image_files[:2000] # Quick subset
    
    if not image_files:
        print("No images found!")
        return
        
    print(f"Generating dynamic labels for {len(image_files)} images...")
    
    success_count = 0
    for idx, img_name in enumerate(image_files):
        img_path = os.path.join(input_dir, img_name)
        if img_name not in telemetry_dict:
            continue
            
        row = telemetry_dict[img_name]
        touch_x = row['touch_x']
        touch_y = row['touch_y']
        
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        h, w = img.shape[:2]
        
        # 1. Detect platform corners dynamically
        results = pose_model.predict(source=img, imgsz=640, conf=0.5, verbose=False)
        result = results[0]
        
        if result.keypoints is None or len(result.keypoints.xy) == 0:
            continue
            
        kpts = result.keypoints.xy[0].cpu().numpy()
        if len(kpts) < 4:
            continue
            
        # 2. Compute dynamic Homography for this specific frame
        ordered_kpts = order_corners(kpts[:4])
        
        # M maps MM -> Pixels directly
        M, _ = cv2.findHomography(BOARD_CORNERS_MM, ordered_kpts)
        if M is None:
            continue
            
        # 3. Project true target markers from MM to Pixels
        target_pts_mm = np.array([TARGETS_PHYSICAL_MM], dtype=np.float32) # 1x4x2
        target_pts_pixel = cv2.perspectiveTransform(target_pts_mm, M)[0]
        
        # 4. Project true ball from MM to Pixels
        ball_x_mm = touch_x + (PLATFORM_W / 2.0)
        ball_y_mm = touch_y + (PLATFORM_H / 2.0)
        ball_pt_mm = np.array([[[ball_x_mm, ball_y_mm]]], dtype=np.float32)
        ball_pt_pixel = cv2.perspectiveTransform(ball_pt_mm, M)[0][0]
        
        # 5. Pack all 5 points
        pts_orig = np.array([
            [target_pts_pixel[0][0], target_pts_pixel[0][1]], # Target 1
            [target_pts_pixel[1][0], target_pts_pixel[1][1]], # Target 2
            [target_pts_pixel[2][0], target_pts_pixel[2][1]], # Target 3
            [target_pts_pixel[3][0], target_pts_pixel[3][1]], # Target 4
            [ball_pt_pixel[0], ball_pt_pixel[1]]              # Ball
        ], dtype=np.float32)
        
        # 6. Apply random warp augmentation
        H_warp = generate_random_warp(w, h)
        img_warped = cv2.warpPerspective(img, H_warp, (w, h), borderMode=cv2.BORDER_REPLICATE)
        
        pts_orig_reshaped = np.array([pts_orig]) # 1x5x2
        pts_warped = cv2.perspectiveTransform(pts_orig_reshaped, H_warp)[0]
        
        # 7. Generate YOLO labels
        labels = []
        for i in range(4):
            mcx, mcy = pts_warped[i]
            mw, mh = 20.0, 20.0 # Bounding box size for marker
            mcx_norm = max(0, min(1, mcx / w))
            mcy_norm = max(0, min(1, mcy / h))
            labels.append(f"{i+1} {mcx_norm:.6f} {mcy_norm:.6f} {mw/w:.6f} {mh/h:.6f}")
            
        bcx, bcy = pts_warped[4]
        bw, bh = 30.0, 30.0
        bcx_norm = max(0, min(1, bcx / w))
        bcy_norm = max(0, min(1, bcy / h))
        labels.append(f"0 {bcx_norm:.6f} {bcy_norm:.6f} {bw/w:.6f} {bh/h:.6f}")
            
        # Save
        out_img_path = os.path.join(out_images_dir, img_name)
        out_txt_path = os.path.join(out_labels_dir, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))
        
        cv2.imwrite(out_img_path, img_warped)
        with open(out_txt_path, 'w') as f:
            f.write('\n'.join(labels) + '\n')
            
        success_count += 1
        if success_count % 100 == 0:
            print(f"Generated {success_count} dynamic labeled images...")
            
    print(f"Done! Successfully generated {success_count} images.")
    
    yaml_content = f"""path: /content/ball_balance_video_controlled/host_software/ml_vision/data/06_dynamic_warped_dataset
train: images
val: images

names:
  0: ball
  1: green_target
  2: red_target
  3: black_target
  4: grey_target
"""
    with open(os.path.join(output_dir, "dataset.yaml"), "w") as f:
        f.write(yaml_content)
    print("Created dataset.yaml for YOLO.")

if __name__ == '__main__':
    main()
