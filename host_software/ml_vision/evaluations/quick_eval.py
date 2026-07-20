import os
import argparse
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
import json
import sys

# Ensure training module is in path to import BallDataset
script_dir = os.path.dirname(os.path.abspath(__file__))
training_dir = os.path.abspath(os.path.join(script_dir, '../training'))
if training_dir not in sys.path:
    sys.path.append(training_dir)

from ball_dataset import BallDataset

def main():
    parser = argparse.ArgumentParser(description="Quick Metric Evaluation for Partially Trained Model")
    parser.add_argument("--data_dir", default="../data/02_silver", help="Path to data directory")
    parser.add_argument("--model_path", default="../models/resnet18_expert_tracker/expert_tracker_latest.pth", help="Path to the model checkpoint")
    parser.add_argument("--num_samples", type=int, default=500, help="Number of random samples to evaluate quickly")
    args = parser.parse_args()

    print(f"Loading partially trained model from: {args.model_path}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.model_path):
        print(f"ERROR: Checkpoint not found at {args.model_path}. Is the model currently training?")
        return

    # Initialize model architecture
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    # Load weights
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    # Load dataset
    csv_path = os.path.join(args.data_dir, 'labels_sequential.csv')
    images_dir = os.path.join(args.data_dir, 'images')
    
    test_transform = transforms.Compose([
        transforms.Resize((240, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    full_dataset = BallDataset(csv_file=csv_path, root_dir=images_dir, transform=test_transform)
    
    # Pick random indices for quick evaluation
    total_samples = min(args.num_samples, len(full_dataset))
    indices = random.sample(range(len(full_dataset)), total_samples)
    subset = Subset(full_dataset, indices)
    
    # Large batch size for speed since we aren't training
    loader = DataLoader(subset, batch_size=128, shuffle=False, num_workers=0)
    
    MAX_BOUND = 200.0 # From ball_dataset.py

    all_preds_x = []
    all_preds_y = []
    all_targets_x = []
    all_targets_y = []

    print(f"Running fast inference on {total_samples} random frames...")
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            # De-normalize coordinates
            preds_mm = outputs.cpu().numpy() * MAX_BOUND
            targs_mm = targets.cpu().numpy() * MAX_BOUND
            
            all_preds_x.extend(preds_mm[:, 0])
            all_preds_y.extend(preds_mm[:, 1])
            all_targets_x.extend(targs_mm[:, 0])
            all_targets_y.extend(targs_mm[:, 1])

    # Calculate Metrics
    preds_x = np.array(all_preds_x)
    preds_y = np.array(all_preds_y)
    targs_x = np.array(all_targets_x)
    targs_y = np.array(all_targets_y)
    
    error_x = preds_x - targs_x
    error_y = preds_y - targs_y
    euclidean_error = np.sqrt(error_x**2 + error_y**2)
    
    # Match the exact format of evaluation_metrics.json
    metrics = {
        "MAE_X_mm": float(np.mean(np.abs(error_x))),
        "MAE_Y_mm": float(np.mean(np.abs(error_y))),
        "RMSE_X_mm": float(np.sqrt(np.mean(error_x**2))),
        "RMSE_Y_mm": float(np.sqrt(np.mean(error_y**2))),
        "Mean_Euclidean_Error_mm": float(np.mean(euclidean_error)),
        "Max_Euclidean_Error_mm": float(np.max(euclidean_error)),
        "95th_Percentile_Error_mm": float(np.percentile(euclidean_error, 95))
    }
    
    print("\n--- Quick Evaluation Metrics (Millimeters) ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.2f} mm")
        
    project_dir = os.path.dirname(os.path.abspath(args.model_path))
    output_path = os.path.join(project_dir, 'quick_evaluation_metrics.json')
    
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print(f"\nSaved quick metrics to {output_path}")

if __name__ == '__main__':
    main()
