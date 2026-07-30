import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import json
import time
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
training_dir = os.path.abspath(os.path.join(script_dir, '../training'))
if training_dir not in sys.path:
    sys.path.append(training_dir)

import argparse
from ball_pixel_dataset import BallPixelDataset
from basic_cnn import BasicCNN

def main():
    parser = argparse.ArgumentParser(description="Evaluate Custom CNN 2D Expert Tracker")
    parser.add_argument("--data_dir", default="../../data/02_silver", help="Path to data directory")
    parser.add_argument("--csv_name", default="yolo_features.csv", help="Name of the CSV labels file")
    parser.add_argument("--model_path", required=True, help="Path to the trained .pth file (e.g. models/cnn_2d_tracker/cnn_2d_tracker_v1/expert_tracker_best.pth)")
    args = parser.parse_args()

    print("Initializing Evaluation Script for Custom CNN Tracker...")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = os.path.abspath(args.model_path)
    project_dir = os.path.dirname(model_path)
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}. Train it first.")
        return
        
    # 1. Initialize model
    model = BasicCNN()
    img_size = (240, 320)
    
    # Load weights safely
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    
    # 2. Load the test subset data
    data_dir = os.path.abspath(args.data_dir)
    csv_path = os.path.join(data_dir, args.csv_name)
    images_dir = os.path.join(data_dir, 'images')
    
    print(f"Loading dataset from: {csv_path}")
    test_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    full_dataset = BallPixelDataset(csv_file=csv_path, root_dir=images_dir, transform=test_transform)
    
    # Test on the last 20% of the dataset sequentially to avoid leakage
    indices = list(range(len(full_dataset)))
    train_size = int(0.8 * len(indices))
    test_indices = indices[train_size:]
    
    test_dataset = Subset(full_dataset, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    print(f"Evaluating on {len(test_dataset)} unseen test frames.")
    
    PLATFORM_W, PLATFORM_H = 187.5, 142.0
    
    all_preds_x = []
    all_preds_y = []
    all_targets_x = []
    all_targets_y = []
    inference_times_ms = []
    
    print("Running inference...")
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            
            t0 = time.perf_counter()
            outputs = model(inputs)
            t1 = time.perf_counter()
            
            batch_size = inputs.size(0)
            time_per_frame_ms = ((t1 - t0) / batch_size) * 1000.0
            for _ in range(batch_size):
                inference_times_ms.append(time_per_frame_ms)
            
            # De-normalize [-1, 1] back to [0, PLATFORM_W] and [0, PLATFORM_H]
            outputs_np = outputs.cpu().numpy()
            targets_np = targets.cpu().numpy()
            
            preds_x = (outputs_np[:, 0] + 1.0) * (PLATFORM_W / 2.0)
            preds_y = (outputs_np[:, 1] + 1.0) * (PLATFORM_H / 2.0)
            
            targs_x = (targets_np[:, 0] + 1.0) * (PLATFORM_W / 2.0)
            targs_y = (targets_np[:, 1] + 1.0) * (PLATFORM_H / 2.0)
            
            all_preds_x.extend(preds_x)
            all_preds_y.extend(preds_y)
            all_targets_x.extend(targs_x)
            all_targets_y.extend(targs_y)
            
    # Calculate Metrics
    preds_x = np.array(all_preds_x)
    preds_y = np.array(all_preds_y)
    targs_x = np.array(all_targets_x)
    targs_y = np.array(all_targets_y)
    
    # R-squared calculation (Mode Collapse check)
    ss_res_x = np.sum((targs_x - preds_x) ** 2)
    ss_tot_x = np.sum((targs_x - np.mean(targs_x)) ** 2)
    r2_x = 1 - (ss_res_x / (ss_tot_x + 1e-8))
    
    ss_res_y = np.sum((targs_y - preds_y) ** 2)
    ss_tot_y = np.sum((targs_y - np.mean(targs_y)) ** 2)
    r2_y = 1 - (ss_res_y / (ss_tot_y + 1e-8))
    
    error_x = preds_x - targs_x
    error_y = preds_y - targs_y
    euclidean_error = np.sqrt(error_x**2 + error_y**2)
    inference_times_ms = np.array(inference_times_ms)
    
    metrics = {
        "MAE_X_mm": float(np.mean(np.abs(error_x))),
        "MAE_Y_mm": float(np.mean(np.abs(error_y))),
        "RMSE_X_mm": float(np.sqrt(np.mean(error_x**2))),
        "RMSE_Y_mm": float(np.sqrt(np.mean(error_y**2))),
        "R2_Score_X": float(r2_x),
        "R2_Score_Y": float(r2_y),
        "Pred_StdDev_X_mm": float(np.std(preds_x)),
        "Pred_StdDev_Y_mm": float(np.std(preds_y)),
        "Mean_Euclidean_Error_mm": float(np.mean(euclidean_error)),
        "Max_Euclidean_Error_mm": float(np.max(euclidean_error)),
        "95th_Percentile_Error_mm": float(np.percentile(euclidean_error, 95)),
        "Mean_Inference_Time_ms": float(np.mean(inference_times_ms)),
        "Max_Inference_Time_ms": float(np.max(inference_times_ms)),
        "FPS_Estimate": float(1000.0 / np.mean(inference_times_ms))
    }
    
    print("\n--- Custom CNN Evaluation Metrics (Millimeters) ---")
    for k, v in metrics.items():
        if "R2" in k:
            print(f"{k}: {v:.3f}")
        else:
            print(f"{k}: {v:.2f} mm" if "Time" not in k and "FPS" not in k and "R2" not in k else f"{k}: {v:.2f}")
            
    if r2_x < 0.1 or r2_y < 0.1 or metrics["Pred_StdDev_X_mm"] < 5.0 or metrics["Pred_StdDev_Y_mm"] < 5.0:
        print("\n[WARNING] MODE COLLAPSE DETECTED!")
        print("The R2 Score is ~0 (or negative) AND/OR the Prediction StdDev is very low.")
        print("This means the neural network is just outputting the center of the board constantly, instead of tracking the ball.")
        
    metrics_path = os.path.join(project_dir, 'evaluation_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    # Plotting X and Y trajectories
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(targs_x, label='Actual X', color='blue', alpha=0.7)
    plt.plot(preds_x, label='Predicted X', color='red', linestyle='--', alpha=0.7)
    plt.ylabel('X Position (mm)')
    plt.title('CNN 2D Tracker Test Set Trajectory (X)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(targs_y, label='Actual Y', color='blue', alpha=0.7)
    plt.plot(preds_y, label='Predicted Y', color='red', linestyle='--', alpha=0.7)
    plt.xlabel('Frame Index (Time)')
    plt.ylabel('Y Position (mm)')
    plt.title('CNN 2D Tracker Test Set Trajectory (Y)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(project_dir, 'evaluation_trajectory.png')
    plt.savefig(plot_path)
    print(f"\nSaved trajectory plot to {plot_path}")
    print(f"Saved metrics to {metrics_path}")
 
if __name__ == '__main__':
    main()
