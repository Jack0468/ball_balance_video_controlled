import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import time
import json
import matplotlib.pyplot as plt
from ultralytics import YOLO

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from core.corrector_mlp import CorrectorMLP
from core.coordinate_math import HomographyProjector

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../data/02_silver", help="Path to telemetry dataset")
    parser.add_argument("--yolo_path", default="../models/platform_and_markers_model/weights/best.pt")
    parser.add_argument("--corrector_path", default="../models/corrector/best_corrector.pth")
    args = parser.parse_args()

    data_dir = os.path.abspath(os.path.join(script_dir, args.data_dir))
    yolo_path = os.path.abspath(os.path.join(script_dir, args.yolo_path))
    corrector_path = os.path.abspath(os.path.join(script_dir, args.corrector_path))
    
    csv_path = os.path.join(data_dir, "labels_sequential.csv")
    images_dir = os.path.join(data_dir, "images")
    
    print(f"Loading YOLO Model from {yolo_path}...")
    yolo_model = YOLO(yolo_path)
    
    print(f"Loading Corrector MLP from {corrector_path}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    corrector = CorrectorMLP().to(device)
    if os.path.exists(corrector_path):
        corrector.load_state_dict(torch.load(corrector_path, map_location=device))
        corrector.eval()
    else:
        print(f"ERROR: Corrector model not found at {corrector_path}. Did you run train_corrector.py?")
        return

    print(f"Loading Telemetry from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Evaluate only on test set (last 20%)
    split_idx = int(0.8 * len(df))
    test_df = df.iloc[split_idx:]
    
    # Initialize Homography Projector
    dst_pts = np.array([
        [-70, 55],
        [70, 55],
        [70, -55],
        [-70, -55]
    ], dtype=np.float32)
    projector = HomographyProjector(dst_pts)
    
    print(f"Evaluating on {len(test_df)} test frames...")
    
    errors_x = []
    errors_y = []
    errors_dist = []
    inference_times_ms = []
    
    processed = 0
    missed_platform = 0
    missed_ball = 0
    
    for idx, row in test_df.iterrows():
        img_path = os.path.join(images_dir, row['image_file'])
        if not os.path.exists(img_path):
            continue
            
        import cv2
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        t0 = time.perf_counter()
        
        # 1. YOLO Inference
        results = yolo_model.predict(source=img, imgsz=640, conf=0.5, verbose=False)
        if not results or len(results) == 0:
            continue
            
        res = results[0]
        if res.boxes is None:
            continue
            
        classes = res.boxes.cls.cpu().numpy()
        boxes = res.boxes.xywh.cpu().numpy()
        
        corners = None
        ball_box = None
        
        for i, cls in enumerate(classes):
            if int(cls) == 0:
                if res.keypoints is not None and len(res.keypoints.xy) > i:
                    kpts = res.keypoints.xy[i].cpu().numpy()
                    if len(kpts) == 4:
                        corners = kpts
            elif int(cls) == 1:
                ball_box = boxes[i]
                
        if corners is None:
            missed_platform += 1
            continue
        if ball_box is None:
            missed_ball += 1
            continue
            
        # 2. MLP Corrector Inference
        homography_x, homography_y = 0.0, 0.0
        if projector.update_homography(corners):
            hx, hy = projector.project_point(ball_box[0], ball_box[1])
            if hx is not None and hy is not None:
                homography_x, homography_y = hx, hy
                
        features = np.array([
            ball_box[0], ball_box[1], ball_box[2], ball_box[3],
            corners[0][0], corners[0][1], corners[1][0], corners[1][1],
            corners[2][0], corners[2][1], corners[3][0], corners[3][1],
            homography_x, homography_y
        ], dtype=np.float32)
        
        # Normalize
        features[0:12:2] /= 640.0
        features[1:12:2] /= 480.0
        features[12:] /= 100.0
        
        features_tensor = torch.tensor(features).unsqueeze(0).to(device)
        with torch.no_grad():
            out = corrector(features_tensor)
            
        pred_x, pred_y = out[0].cpu().numpy()
        
        t1 = time.perf_counter()
        
        true_x = row['touch_x']
        true_y = row['touch_y']
        
        err_x = pred_x - true_x
        err_y = pred_y - true_y
        dist = np.sqrt(err_x**2 + err_y**2)
        
        errors_x.append(err_x)
        errors_y.append(err_y)
        errors_dist.append(dist)
        inference_times_ms.append((t1 - t0) * 1000.0)
        
        processed += 1
        if processed % 1000 == 0:
            print(f"Evaluated {processed}/{len(test_df)} frames...")
            
    print("-" * 50)
    print("Evaluation Complete!")
    print(f"Total Frames: {len(test_df)}")
    print(f"Successfully Processed: {processed}")
    print(f"Missed Platform: {missed_platform}")
    print(f"Missed Ball: {missed_ball}")
    
    if processed == 0:
        print("No valid predictions were made.")
        return
        
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
    
    print("\n--- YOLO + MLP Corrector Evaluation Metrics (Millimeters) ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}")
        
    out_dir = os.path.dirname(corrector_path)
    json_path = os.path.join(out_dir, "evaluation_metrics.json")
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print(f"\nSaved metrics to {json_path}")
    
if __name__ == '__main__':
    main()
