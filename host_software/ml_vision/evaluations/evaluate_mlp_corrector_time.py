import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
import json
import time
import argparse

import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
training_dir = os.path.abspath(os.path.join(script_dir, '../training'))
if training_dir not in sys.path:
    sys.path.append(training_dir)

from train_mlp_corrector_time import MLPCorrectorTime, TimeWindowDataset

def main():
    parser = argparse.ArgumentParser(description="Evaluate MLP Time Corrector")
    parser.add_argument("--csv_path", default="../data/02_silver/session_20260730_174916/cnn_sequential_features.csv", help="Path to cnn_sequential_features.csv")
    parser.add_argument("--model_path", required=True, help="Path to the trained .pth file")
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--future_offset", type=int, default=0)
    args = parser.parse_args()

    print(f"Initializing Evaluation Script for MLP Time Corrector...")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.abspath(args.model_path)
    
    # Actually, data is at VRI_2026/host_software/data.
    # script_dir is VRI_2026/host_software/ml_vision/evaluations
    # So relative to script_dir, data is at ../../data
    # Wait! If the user passed default string ../../data, it was joining to script_dir.
    # Let me just use an absolute path based on __file__ manually to be safe.
    host_software_dir = os.path.abspath(os.path.join(script_dir, '../../'))
    
    # If the user provides an absolute path, os.path.join will just use it. 
    # If relative, we assume it's relative to current working directory (which is VRI_2026)
    csv_path = os.path.abspath(args.csv_path)
    project_dir = os.path.dirname(model_path)
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}.")
        return
        
    # 1. Initialize model
    model = MLPCorrectorTime(window_size=args.window_size)
    
    # Load weights safely
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    
    # 2. Load the test subset data
    print(f"Loading dataset from: {csv_path}")
    
    # Use future_offset as it was trained
    full_dataset = TimeWindowDataset(csv_file=csv_path, window_size=args.window_size, future_offset=args.future_offset)
    
    if len(full_dataset) == 0:
        print("Error: No valid sequences found in dataset!")
        return
        
    # Test on the last 20% of the dataset
    indices = list(range(len(full_dataset)))
    train_size = int(0.8 * len(indices))
    test_indices = indices[train_size:]
    
    test_dataset = Subset(full_dataset, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    print(f"Evaluating on {len(test_dataset)} unseen test sequences.")
    
    all_preds = []
    all_targets = []
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
            
            # Outputs and targets are natively in mm and mm/s
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            
    preds = np.concatenate(all_preds, axis=0)
    targs = np.concatenate(all_targets, axis=0)
    
    # Calculate Metrics
    # Output indices: 0: touch_x, 1: touch_y
    error = preds - targs
    
    mae = np.mean(np.abs(error), axis=0)
    rmse = np.sqrt(np.mean(error**2, axis=0))
    p95 = np.percentile(np.abs(error), 95, axis=0)
    
    inference_times_ms = np.array(inference_times_ms)
    
    metrics = {
        "Position_X_MAE_mm": float(mae[0]),
        "Position_Y_MAE_mm": float(mae[1]),
        "Position_X_RMSE_mm": float(rmse[0]),
        "Position_Y_RMSE_mm": float(rmse[1]),
        "Position_X_95th_mm": float(p95[0]),
        "Position_Y_95th_mm": float(p95[1]),
        "Mean_Inference_Time_ms": float(np.mean(inference_times_ms)),
        "FPS_Estimate": float(1000.0 / np.mean(inference_times_ms))
    }
    
    print("\n--- Evaluation Metrics ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}")
        
    metrics_path = os.path.join(project_dir, 'evaluation_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    # Plotting X and Y trajectories
    # We will just plot the first 150 frames of the test set for clear visibility
    plot_len = min(150, len(preds))
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(targs[:plot_len, 0], label='Actual X', color='blue', alpha=0.7, linewidth=2)
    plt.plot(preds[:plot_len, 0], label='Predicted X', color='red', linestyle='--', alpha=0.9, linewidth=2)
    plt.ylabel('X Position (mm)')
    plt.title('Unseen Test Trajectory - X Position')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(targs[:plot_len, 1], label='Actual Y', color='blue', alpha=0.7, linewidth=2)
    plt.plot(preds[:plot_len, 1], label='Predicted Y', color='red', linestyle='--', alpha=0.9, linewidth=2)
    plt.xlabel('Frame Index')
    plt.ylabel('Y Position (mm)')
    plt.title('Unseen Test Trajectory - Y Position')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(project_dir, 'evaluation_trajectory.png')
    plt.savefig(plot_path)
    print(f"\nSaved trajectory plot to {plot_path}")
    print(f"Saved metrics to {metrics_path}")

if __name__ == '__main__':
    main()
