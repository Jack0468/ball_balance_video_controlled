import os
import cv2
import numpy as np
import pandas as pd
import random
import argparse
from ultralytics import YOLO

# Physical dimensions of the purple board (in mm)
PLATFORM_W = 200.0
PLATFORM_H = 146.93


# The physical corners of the board (Top-Left origin)
BOARD_CORNERS_MM = np.array([
    [0.0, 0.0],                     # Top-Left
    [PLATFORM_W, 0.0],              # Top-Right
    [PLATFORM_W, PLATFORM_H],       # Bottom-Right
    [0.0, PLATFORM_H]               # Bottom-Left
], dtype=np.float32)

# Physical coordinates of the 4 target markers (Relative to Top-Left of the Resistive Touchpad)
# The touchpad is 187.5 x 142.0
TOUCHPAD_W = 187.5
TOUCHPAD_H = 142.0

TARGETS_PHYSICAL_MM = np.array([
    [33.0, 26.0],                           # Target 1 (Green, Top-Left)
    [TOUCHPAD_W - 41.0, 53.0],              # Target 2 (Red, Top-Right)
    [TOUCHPAD_W - 13.0, TOUCHPAD_H - 8.0],  # Target 3 (Black, Bottom-Right)
    [69.0, TOUCHPAD_H - 58.0]               # Target 4 (Grey, Bottom-Left)
], dtype=np.float32)

# Calculate the offset of the touchpad relative to the outer purple plate
PAD_OFFSET_X = (PLATFORM_W - TOUCHPAD_W) / 2.0
PAD_OFFSET_Y = (PLATFORM_H - TOUCHPAD_H) / 2.0

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
    # 1. Perspective jitter
    pts1 = np.float32([[0,0], [img_w,0], [img_w,img_h], [0,img_h]])
    jitter_x = int(img_w * 0.15)
    jitter_y = int(img_h * 0.15)
    
    pts2 = np.float32([
        [random.randint(-jitter_x, jitter_x), random.randint(-jitter_y, jitter_y)],
        [img_w + random.randint(-jitter_x, jitter_x), random.randint(-jitter_y, jitter_y)],
        [img_w + random.randint(-jitter_x, jitter_x), img_h + random.randint(-jitter_x, jitter_x)],
        [random.randint(-jitter_x, jitter_x), img_h + random.randint(-jitter_x, jitter_x)]
    ])
    
    H_persp = cv2.getPerspectiveTransform(pts1, pts2)
    
    # 2. 360-degree rotation (Random arbitrary yaw)
    angle = random.uniform(0, 360)
    center = (img_w / 2, img_h / 2)
    M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    H_rot = np.eye(3)
    H_rot[0:2, :] = M_rot
    
    # Combine: Perspective then Rotation
    matrix = H_rot @ H_persp
    return matrix

def format_pose_label(class_id, cx, cy, w, h, keypoints=None, img_w=640, img_h=480):
    """
    Format a YOLOv8-Pose label.
    If keypoints are None, pads with 4 invisible keypoints (0, 0, 0).
    """
    cx_norm = max(0.0, min(1.0, cx / img_w))
    cy_norm = max(0.0, min(1.0, cy / img_h))
    w_norm = max(0.0, min(1.0, w / img_w))
    h_norm = max(0.0, min(1.0, h / img_h))
    
    label = f"{class_id} {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}"
    
    if keypoints is not None:
        for (kx, ky) in keypoints:
            kx_n = max(0.0, min(1.0, kx / img_w))
            ky_n = max(0.0, min(1.0, ky / img_h))
            label += f" {kx_n:.6f} {ky_n:.6f} 2" # visibility=2 (visible)
    else:
        # Pad with 4 invisible keypoints for non-platform classes
        for _ in range(4):
            label += " 0.000000 0.000000 0"
            
    return label

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dataset", type=str, default="02_silver", help="Name of dataset in data/ directory")
    parser.add_argument("--csv_name", type=str, default="labels_sequential.csv", help="Name of the telemetry CSV file")
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.abspath(os.path.join(script_dir, f"../data/{args.input_dataset}/images"))
    telemetry_path = os.path.abspath(os.path.join(script_dir, f"../data/{args.input_dataset}/{args.csv_name}"))
    
    output_dir = os.path.abspath(os.path.join(script_dir, f"../data/{args.input_dataset}_unified_pose"))
    out_images_dir = os.path.join(output_dir, "images")
    out_labels_dir = os.path.join(output_dir, "labels")
    
    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_labels_dir, exist_ok=True)
    
    print("Loading pre-trained YOLO-Pose platform model...")
    try:
        pose_model = YOLO(os.path.abspath(os.path.join(script_dir, "../models/platform_pose_model/weights/best.pt")))
    except Exception as e:
        print(f"Failed to load platform pose model: {e}")
        return
        
    print(f"Loading telemetry from {telemetry_path}...")
    if not os.path.exists(telemetry_path):
        print(f"ERROR: Telemetry file {telemetry_path} not found!")
        return
        
    df = pd.read_csv(telemetry_path)
    telemetry_dict = {str(row['image_file']): row for _, row in df.iterrows()}
        
    image_files = [f for f in os.listdir(input_dir) if f.endswith('.jpg') or f.endswith('.png')]
    if not image_files:
        print(f"No images found in {input_dir}!")
        return
        
    print(f"Generating unified dynamic labels for {len(image_files)} images...")
    
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
        M, _ = cv2.findHomography(BOARD_CORNERS_MM, ordered_kpts)
        if M is None:
            continue
            
        # 3. Project true target markers from MM to Pixels
        # Shift targets from Touchpad Space to Plate Space
        target_pts_mm = np.array([TARGETS_PHYSICAL_MM], dtype=np.float32)
        target_pts_mm[0, :, 0] += PAD_OFFSET_X
        target_pts_mm[0, :, 1] += PAD_OFFSET_Y
        
        target_pts_pixel = cv2.perspectiveTransform(target_pts_mm, M)[0]
        
        # 4. Project true ball from MM to Pixels
        ball_x_mm = touch_x + (PLATFORM_W / 2.0)
        ball_y_mm = touch_y + (PLATFORM_H / 2.0)
        ball_pt_mm = np.array([[[ball_x_mm, ball_y_mm]]], dtype=np.float32)
        ball_pt_pixel = cv2.perspectiveTransform(ball_pt_mm, M)[0][0]
        
        # 5. Apply random warp augmentation
        H_warp = generate_random_warp(w, h)
        img_warped = cv2.warpPerspective(img, H_warp, (w, h), borderMode=cv2.BORDER_REPLICATE)
        
        # Warp Platform Keypoints
        kpts_reshaped = np.array([ordered_kpts])
        warped_kpts = cv2.perspectiveTransform(kpts_reshaped, H_warp)[0]
        
        # Determine Platform Bounding Box from warped keypoints
        plat_x_min, plat_y_min = np.min(warped_kpts, axis=0)
        plat_x_max, plat_y_max = np.max(warped_kpts, axis=0)
        plat_cx = (plat_x_min + plat_x_max) / 2.0
        plat_cy = (plat_y_min + plat_y_max) / 2.0
        plat_w = plat_x_max - plat_x_min
        plat_h = plat_y_max - plat_y_min
        
        # Warp Ball and Targets
        ball_target_pts = np.vstack([ball_pt_pixel, target_pts_pixel])
        bt_reshaped = np.array([ball_target_pts])
        warped_bt = cv2.perspectiveTransform(bt_reshaped, H_warp)[0]
        warped_ball = warped_bt[0]
        warped_targets = warped_bt[1:]
        
        # 6. Generate YOLO-Pose labels
        labels = []
        
        # Class 0: Platform (with 4 keypoints)
        labels.append(format_pose_label(0, plat_cx, plat_cy, plat_w, plat_h, keypoints=warped_kpts, img_w=w, img_h=h))
        
        # Class 1: Ball (no keypoints)
        bw, bh = 30.0, 30.0
        labels.append(format_pose_label(1, warped_ball[0], warped_ball[1], bw, bh, keypoints=None, img_w=w, img_h=h))
        
        # Classes 2-5: Targets (no keypoints)
        for i, target_px in enumerate(warped_targets):
            tw, th = 20.0, 20.0
            class_id = i + 2
            labels.append(format_pose_label(class_id, target_px[0], target_px[1], tw, th, keypoints=None, img_w=w, img_h=h))
            
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
    
    yaml_content = f"""path: /content/ball_balance_video_controlled/host_software/ml_vision/data/{args.input_dataset}_unified_pose
train: images
val: images

kpt_shape: [4, 3] # 4 keypoints, 3 values (x, y, visibility)

names:
  0: platform
  1: ball
  2: green_target
  3: red_target
  4: black_target
  5: grey_target
"""
    with open(os.path.join(output_dir, "dataset.yaml"), "w") as f:
        f.write(yaml_content)
    print("Created dataset.yaml for YOLO-Pose.")

if __name__ == '__main__':
    main()
