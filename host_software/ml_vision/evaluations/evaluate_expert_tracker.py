import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np
import json

import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
training_dir = os.path.abspath(os.path.join(script_dir, '../training'))
if training_dir not in sys.path:
    sys.path.append(training_dir)

from ball_dataset import BallDataset

def main():
    print("Initializing Evaluation Script for Expert Tracker (Subset)...")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(script_dir, '../models/resnet18_expert_tracker_subset'))
    model_path = os.path.join(project_dir, 'expert_tracker_subset_best.pth')
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}. Train it first.")
        return
        
    # 1. Initialize model
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    # Load weights safely
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    
    # 2. Load the test subset data
    data_dir = os.path.abspath(os.path.join(script_dir, '../data/02_silver'))
    csv_path = os.path.join(data_dir, 'labels.csv')
    images_dir = os.path.join(data_dir, 'images')
    
    print(f"Loading dataset from: {csv_path}")
    full_dataset = BallDataset(csv_file=csv_path, root_dir=images_dir)
    
    # Replicate subset logic from training script to get the exact test set
    START_INDEX = 15000
    SUBSET_SIZE = 1000
    
    if len(full_dataset) >= START_INDEX + SUBSET_SIZE:
        indices = list(range(START_INDEX, START_INDEX + SUBSET_SIZE))
    else:
        indices = list(range(len(full_dataset)))
        
    train_size = int(0.8 * len(indices))
    test_indices = indices[train_size:]
    
    test_dataset = Subset(full_dataset, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    print(f"Evaluating on {len(test_dataset)} unseen future contiguous frames.")
    
    MAX_BOUND = 200.0 # From ball_dataset.py
    
    all_preds_x = []
    all_preds_y = []
    all_targets_x = []
    all_targets_y = []
    
    print("Running inference...")
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            # De-normalize
            outputs_mm = outputs.cpu().numpy() * MAX_BOUND
            targets_mm = targets.cpu().numpy() * MAX_BOUND
            
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
    
    metrics = {
        "MAE_X_mm": float(np.mean(np.abs(error_x))),
        "MAE_Y_mm": float(np.mean(np.abs(error_y))),
        "RMSE_X_mm": float(np.sqrt(np.mean(error_x**2))),
        "RMSE_Y_mm": float(np.sqrt(np.mean(error_y**2))),
        "Mean_Euclidean_Error_mm": float(np.mean(euclidean_error)),
        "Max_Euclidean_Error_mm": float(np.max(euclidean_error)),
        "95th_Percentile_Error_mm": float(np.percentile(euclidean_error, 95))
    }
    
    print("\n--- Evaluation Metrics (Millimeters) ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.2f} mm")
        
    metrics_path = os.path.join(project_dir, 'evaluation_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    # Plotting X and Y trajectories
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(targs_x, label='Actual X', color='blue', alpha=0.7)
    plt.plot(preds_x, label='Predicted X', color='red', linestyle='--', alpha=0.7)
    plt.ylabel('X Position (mm)')
    plt.title('Contiguous Test Set Trajectory (X)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(targs_y, label='Actual Y', color='blue', alpha=0.7)
    plt.plot(preds_y, label='Predicted Y', color='red', linestyle='--', alpha=0.7)
    plt.xlabel('Frame Index (Time)')
    plt.ylabel('Y Position (mm)')
    plt.title('Contiguous Test Set Trajectory (Y)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(project_dir, 'evaluation_trajectory.png')
    plt.savefig(plot_path)
    print(f"\nSaved trajectory plot to {plot_path}")
    print(f"Saved metrics to {metrics_path}")

if __name__ == '__main__':
    main()
