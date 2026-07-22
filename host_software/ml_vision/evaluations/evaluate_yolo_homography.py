import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import pandas as pd
import cv2
import argparse
import sys
import json
import time
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Add parent directory to path to import core modules
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from core.coordinate_math import HomographyProjector

def get_platform_corners_and_ball(results):
    corners = None
    ball_center = None
    
    if not results or len(results) == 0:
        return None, None
        
    res = results[0]
    if res.boxes is None:
        return None, None
        
    classes = res.boxes.cls.cpu().numpy()
    boxes = res.boxes.xywh.cpu().numpy()
    
    for i, cls in enumerate(classes):
        if int(cls) == 0: # Platform
            if res.keypoints is not None and len(res.keypoints.xy) > i:
                kpts = res.keypoints.xy[i].cpu().numpy()
                if len(kpts) == 4:
                    corners = kpts
        elif int(cls) == 1: # Ball
            bx, by, bw, bh = boxes[i]
            ball_center = (bx, by)
            
    return corners, ball_center

def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO Pose Homography Accuracy")
    parser.add_argument("--data_dir", default="../data/02_silver", help="Path to telemetry dataset")
    parser.add_argument("--model_path", default="../models/platform_and_markers_model/weights/best.pt", help="Path to YOLO model")
    args = parser.parse_args()

    model_path = os.path.abspath(os.path.join(script_dir, args.model_path))
    data_dir = os.path.abspath(os.path.join(script_dir, args.data_dir))
    
    csv_path = os.path.join(data_dir, "labels_sequential.csv")
    images_dir = os.path.join(data_dir, "images")
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}.")
        return
        
    if not os.path.exists(csv_path):
        print(f"ERROR: Dataset CSV not found at {csv_path}.")
        return

    print(f"Loading YOLO Model from {model_path}...")
    model = YOLO(model_path)
    
    print(f"Loading Telemetry from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # We want to test on the last 20% of the dataset to match expert tracker
    train_size = int(0.8 * len(df))
    df = df.iloc[train_size:]
    print(f"Evaluating on {len(df)} unseen test frames.")
    
    # Destination points for Homography (Platform physical dimensions in mm)
    dst_pts = np.array([
        [-70, 55],
        [70, 55],
        [70, -55],
        [-70, -55]
    ], dtype=np.float32)
    
    projector = HomographyProjector(dst_pts)
    
    errors_x = []
    errors_y = []
    errors_dist = []
    inference_times_ms = []
    
    all_preds_x = []
    all_preds_y = []
    all_targets_x = []
    all_targets_y = []
    
    total = len(df)
    processed = 0
    missed_platform = 0
    missed_ball = 0
    
    print("Starting evaluation...")
    for idx, row in df.iterrows():
        img_name = row['image_file']
        true_x = row['touch_x']
        true_y = row['touch_y']
        
        img_path = os.path.join(images_dir, img_name)
        if not os.path.exists(img_path):
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        t0 = time.perf_counter()
        results = model.predict(source=img, imgsz=640, conf=0.5, verbose=False)
        t1 = time.perf_counter()
        
        corners, ball_center = get_platform_corners_and_ball(results)
        
        if corners is None:
            missed_platform += 1
            continue
            
        if ball_center is None:
            missed_ball += 1
            continue
            
        if projector.update_homography(corners):
            px, py = ball_center
            pred_x, pred_y = projector.project_point(px, py)
            
            if pred_x is not None and pred_y is not None:
                err_x = pred_x - true_x
                err_y = pred_y - true_y
                dist = np.sqrt(err_x**2 + err_y**2)
                
                errors_x.append(err_x)
                errors_y.append(err_y)
                errors_dist.append(dist)
                
                all_preds_x.append(pred_x)
                all_preds_y.append(pred_y)
                all_targets_x.append(true_x)
                all_targets_y.append(true_y)
                
                inference_times_ms.append((t1 - t0) * 1000.0)
                
                processed += 1
                
        if processed > 0 and processed % 200 == 0:
            print(f"Processed {processed}/{total} images...")

    print("--------------------------------------------------")
    print(f"Evaluation Complete!")
    print(f"Total Frames: {total}")
    print(f"Successfully Processed: {processed}")
    print(f"Missed Platform: {missed_platform}")
    print(f"Missed Ball: {missed_ball}")
    
    if processed > 0:
        errors_x = np.array(errors_x)
        errors_y = np.array(errors_y)
        errors_dist = np.array(errors_dist)
        inference_times_ms = np.array(inference_times_ms)
        
        metrics = {
            "MAE_X_mm": float(np.mean(np.abs(errors_x))),
            "MAE_Y_mm": float(np.mean(np.abs(errors_y))),
            "RMSE_X_mm": float(np.sqrt(np.mean(errors_x**2))),
            "RMSE_Y_mm": float(np.sqrt(np.mean(errors_y**2))),
            "Mean_Euclidean_Error_mm": float(np.mean(errors_dist)),
            "Max_Euclidean_Error_mm": float(np.max(errors_dist)),
            "95th_Percentile_Error_mm": float(np.percentile(errors_dist, 95)),
            "Mean_Inference_Time_ms": float(np.mean(inference_times_ms)),
            "Max_Inference_Time_ms": float(np.max(inference_times_ms)),
            "FPS_Estimate": float(1000.0 / np.mean(inference_times_ms))
        }
        
        print("\n--- Evaluation Metrics (Millimeters) ---")
        for k, v in metrics.items():
            print(f"{k}: {v:.2f} mm")
            
        # Save JSON
        project_dir = os.path.dirname(os.path.dirname(model_path)) # up from weights/
        metrics_path = os.path.join(project_dir, 'evaluation_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
            
        print(f"Saved metrics to {metrics_path}")
        
        # Plotting X and Y trajectories
        plt.figure(figsize=(12, 6))
        
        plt.subplot(2, 1, 1)
        plt.plot(all_targets_x, label='Actual X', color='blue', alpha=0.7)
        plt.plot(all_preds_x, label='Predicted X (YOLO)', color='red', linestyle='--', alpha=0.7)
        plt.ylabel('X Position (mm)')
        plt.title('Contiguous Test Set Trajectory (X)')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(all_targets_y, label='Actual Y', color='blue', alpha=0.7)
        plt.plot(all_preds_y, label='Predicted Y (YOLO)', color='red', linestyle='--', alpha=0.7)
        plt.xlabel('Frame Index (Time)')
        plt.ylabel('Y Position (mm)')
        plt.title('Contiguous Test Set Trajectory (Y)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(project_dir, 'evaluation_trajectory.png')
        plt.savefig(plot_path)
        print(f"Saved trajectory plot to {plot_path}")
        
    else:
        print("No valid predictions were made.")

if __name__ == '__main__':
    main()
