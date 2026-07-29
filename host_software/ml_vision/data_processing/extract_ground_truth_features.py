import os
import sys
import numpy as np
import pandas as pd
import cv2

parent_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from core.coordinate_math import HomographyProjector

def extract_ground_truth_features():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_base = os.path.abspath(os.path.join(script_dir, '..', '..', 'data'))
    
    # We will process both datasets
    datasets = [
        {
            "gold_dir": os.path.join(data_base, "02_silver", "session_20260728_102908"),
            "silver_csv": os.path.join(data_base, "02_silver", "session_20260728_102908", "labels_normalized.csv")
        },
        {
            "gold_dir": os.path.join(data_base, "03_gold", "images_iphone"),
            "silver_csv": os.path.join(data_base, "02_silver", "images_iphone", "labels_normalized.csv")
        }
    ]
    
    dst_pts = np.array([
        [-70, 55],
        [70, 55],
        [70, -55],
        [-70, -55]
    ], dtype=np.float32)
    projector = HomographyProjector(dst_pts)
    
    IMG_WIDTH = 640.0
    IMG_HEIGHT = 480.0
    
    for ds in datasets:
        gold_dir = ds["gold_dir"]
        silver_csv = ds["silver_csv"]
        
        if not os.path.exists(gold_dir):
            print(f"Directory not found: {gold_dir}")
            continue
            
        if not os.path.exists(silver_csv):
            print(f"Silver CSV not found: {silver_csv}")
            continue
            
        print(f"\nProcessing {gold_dir} ...")
        
        # Load telemetry
        df_telemetry = pd.read_csv(silver_csv)
        
        # Determine train/test split (80/20) based on the sequential order of telemetry
        split_idx = int(0.8 * len(df_telemetry))
        df_telemetry['split'] = 'train'
        df_telemetry.loc[split_idx:, 'split'] = 'test'
        
        # Create a sequential lookup
        telemetry_lookup = {}
        for idx, row in df_telemetry.iterrows():
            base_img = os.path.splitext(row['image_file'])[0]
            data = {
                'image_file': row['image_file'],
                'touch_x': row['touch_x'],
                'touch_y': row['touch_y'],
                'split': row['split']
            }
            telemetry_lookup[base_img] = data
            padded_idx = f"{idx:04d}"
            telemetry_lookup[padded_idx] = data
        
        labels_dir = os.path.join(gold_dir, "labels")
        if not os.path.exists(labels_dir):
            print(f"Labels directory not found: {labels_dir}")
            continue
            
        features = []
        
        for label_file in os.listdir(labels_dir):
            if not label_file.endswith(".txt"):
                continue
                
            basename = os.path.splitext(label_file)[0]
            if basename not in telemetry_lookup:
                continue
                
            telemetry_data = telemetry_lookup[basename]
            image_file = telemetry_data['image_file']
            
            with open(os.path.join(labels_dir, label_file), 'r') as f:
                lines = f.readlines()
                
            platform_box = None
            platform_kpts = None
            ball_box = None
            
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                    
                cls_id = int(parts[0])
                if cls_id == 0:  # Platform
                    # class x y w h kpt0_x kpt0_y conf0 kpt1_x kpt1_y conf1 ...
                    platform_box = [float(p) for p in parts[1:5]]
                    # Parse keypoints if they exist
                    if len(parts) >= 17:
                        # kpt0_x, kpt0_y, conf0, kpt1_x, kpt1_y, conf1, ...
                        kpts = []
                        for i in range(4):
                            idx = 5 + (i * 3)
                            kx = float(parts[idx]) * IMG_WIDTH
                            ky = float(parts[idx+1]) * IMG_HEIGHT
                            kpts.append([kx, ky])
                        platform_kpts = np.array(kpts)
                elif cls_id == 1:  # Ball
                    ball_box = [float(p) for p in parts[1:5]]
                    
            if platform_kpts is not None and ball_box is not None:
                # Convert normalized ball box to pixel coordinates
                ball_x = ball_box[0] * IMG_WIDTH
                ball_y = ball_box[1] * IMG_HEIGHT
                ball_w = ball_box[2] * IMG_WIDTH
                ball_h = ball_box[3] * IMG_HEIGHT
                
                homography_x, homography_y = 0.0, 0.0
                if projector.update_homography(platform_kpts):
                    hx, hy = projector.project_point(ball_x, ball_y)
                    if hx is not None and hy is not None:
                        homography_x, homography_y = hx, hy
                        
                features.append({
                    'image_file': image_file,
                    'split': telemetry_data['split'],
                    'ball_x': ball_x,
                    'ball_y': ball_y,
                    'ball_w': ball_w,
                    'ball_h': ball_h,
                    'kpt0_x': platform_kpts[0][0], 'kpt0_y': platform_kpts[0][1],
                    'kpt1_x': platform_kpts[1][0], 'kpt1_y': platform_kpts[1][1],
                    'kpt2_x': platform_kpts[2][0], 'kpt2_y': platform_kpts[2][1],
                    'kpt3_x': platform_kpts[3][0], 'kpt3_y': platform_kpts[3][1],
                    'homography_x': homography_x,
                    'homography_y': homography_y,
                    'touch_x': telemetry_data['touch_x'],
                    'touch_y': telemetry_data['touch_y']
                })
                
        if features:
            out_csv = os.path.join(gold_dir, "ground_truth_features.csv")
            out_df = pd.DataFrame(features)
            # Sort by image_file to maintain determinism
            out_df = out_df.sort_values(by='image_file')
            out_df.to_csv(out_csv, index=False)
            print(f"Extracted {len(features)} ground truth features to {out_csv}")
        else:
            print(f"No fully labeled frames (platform + ball) found with telemetry in {gold_dir}")

if __name__ == '__main__':
    extract_ground_truth_features()
