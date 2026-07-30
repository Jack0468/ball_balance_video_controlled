import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms
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
from ball_dataset import BallDataset

def main():
    parser = argparse.ArgumentParser(description="Evaluate ResNet 2D Tracker V1")
    parser.add_argument("--data_dir", default="../../data/02_silver/cropped_yolo", help="Path to cropped data directory")
    parser.add_argument("--csv_name", default="labels_normalized.csv", help="Name of the CSV labels file")
    parser.add_argument("--model_path", required=True, help="Path to the trained .pth file (e.g. models/resnet18_2d_tracker_v1/expert_tracker_best.pth)")
    parser.add_argument("--arch", type=str, default="resnet18", choices=["resnet18", "resnet50"], help="Architecture to use")
    args = parser.parse_args()

    print(f"Initializing Evaluation Script for ResNet 2D Tracker V1 ({args.arch})...")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = os.path.abspath(args.model_path)
    project_dir = os.path.dirname(model_path)
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}. Train it first.")
        return
        
    # 1. Initialize model
    if args.arch == "resnet18":
        model = models.resnet18(weights=None)
        img_size = (240, 320)
    elif args.arch == "resnet50":
        model = models.resnet50(weights=None)
        img_size = (480, 640)
        
    # Load weights safely and determine fc structure dynamically
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
    
    # Check if fc layer is a Sequential block in the saved checkpoint
    has_sequential_fc = any(k.startswith('fc.1.') for k in state_dict.keys())
    
    num_ftrs = model.fc.in_features
    if has_sequential_fc:
        print("[INFO] Loading model with Sequential Dropout regression head...")
        model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_ftrs, 2)
        )
    else:
        print("[INFO] Loading model with Linear regression head...")
        model.fc = nn.Linear(num_ftrs, 2)
        
    model.load_state_dict(state_dict)
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
    
    full_dataset = BallDataset(csv_file=csv_path, root_dir=images_dir, transform=test_transform)
    
    # We want to test on the last 20% of the dataset sequentially to prevent temporal leakage
    indices = list(range(len(full_dataset)))
    train_size = int(0.8 * len(indices))
    test_indices = indices[train_size:]
    
    test_dataset = Subset(full_dataset, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    print(f"Evaluating on {len(test_dataset)} unseen test frames.")
    
    MAX_X_BOUND, MAX_Y_BOUND = 93.75, 71.0 # True physical plate bounds
    
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
            
            # De-normalize coordinates
            outputs_mm = outputs.cpu().numpy() * np.array([MAX_X_BOUND, MAX_Y_BOUND])
            targets_mm = targets.cpu().numpy() * np.array([MAX_X_BOUND, MAX_Y_BOUND])
            
            all_preds_x.extend(outputs_mm[:, 0])
            all_preds_y.extend(outputs_mm[:, 1])
            all_targets_x.extend(targets_mm[:, 0])
            all_targets_y.extend(targets_mm[:, 1])
            
    # Calculate Metrics
    preds_x = np.array(all_preds_x)
    preds_y = np.array(all_preds_y)
    targs_x = np.array(all_targets_x)
    targs_y = np.array(all_targets_y)
    
    error_x = preds_x - targs_x
    error_y = preds_y - targs_y
    euclidean_error = np.sqrt(error_x**2 + error_y**2)
    inference_times_ms = np.array(inference_times_ms)
    
    metrics = {
        "MAE_X_mm": float(np.mean(np.abs(error_x))),
        "MAE_Y_mm": float(np.mean(np.abs(error_y))),
        "RMSE_X_mm": float(np.sqrt(np.mean(error_x**2))),
        "RMSE_Y_mm": float(np.sqrt(np.mean(error_y**2))),
        "Mean_Euclidean_Error_mm": float(np.mean(euclidean_error)),
        "Max_Euclidean_Error_mm": float(np.max(euclidean_error)),
        "95th_Percentile_Error_mm": float(np.percentile(euclidean_error, 95)),
        "Mean_Inference_Time_ms": float(np.mean(inference_times_ms)),
        "Max_Inference_Time_ms": float(np.max(inference_times_ms)),
        "FPS_Estimate": float(1000.0 / np.mean(inference_times_ms))
    }
    
    print("\n--- Evaluation Metrics (Millimeters) ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.2f} mm" if "Time" not in k and "FPS" not in k else f"{k}: {v:.2f}")
        
    metrics_path = os.path.join(project_dir, 'evaluation_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    # Plotting X and Y trajectories
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(targs_x, label='Actual X', color='blue', alpha=0.7)
    plt.plot(preds_x, label='Predicted X', color='red', linestyle='--', alpha=0.7)
    plt.ylabel('X Position (mm)')
    plt.title('ResNet 2D Tracker Test Set Trajectory (X)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(targs_y, label='Actual Y', color='blue', alpha=0.7)
    plt.plot(preds_y, label='Predicted Y', color='red', linestyle='--', alpha=0.7)
    plt.xlabel('Frame Index (Time)')
    plt.ylabel('Y Position (mm)')
    plt.title('ResNet 2D Tracker Test Set Trajectory (Y)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(project_dir, 'evaluation_trajectory.png')
    plt.savefig(plot_path)
    print(f"\nSaved trajectory plot to {plot_path}")
    print(f"Saved metrics to {metrics_path}")

if __name__ == '__main__':
    main()
