import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms
import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse

import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
training_dir = os.path.abspath(os.path.join(script_dir, '../training'))
if training_dir not in sys.path:
    sys.path.append(training_dir)

from ball_dataset import BallDataset

def find_worst_predictions(model_path, data_dir, output_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model = model.to(device)
    model.eval()
    
    csv_path = os.path.join(data_dir, 'labels_sequential.csv')
    images_dir = os.path.join(data_dir, 'images')
    
    print(f"Loading dataset from: {csv_path}")
    test_transform = transforms.Compose([
        transforms.Resize((240, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    full_dataset = BallDataset(csv_file=csv_path, root_dir=images_dir, transform=test_transform)
    
    # Test on a random subset of 2000 images from the entire dataset for speed
    import random
    indices = list(range(len(full_dataset)))
    random.shuffle(indices)
    test_indices = indices[:2000]
    
    test_dataset = Subset(full_dataset, test_indices)
    
    # We use batch size 64 for speed
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    MAX_BOUND = 200.0
    
    print("Running inference to find worst errors...")
    errors = []
    
    with torch.no_grad():
        global_idx = 0
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            out_np = outputs.cpu().numpy() * MAX_BOUND
            targ_np = targets.cpu().numpy() * MAX_BOUND
            
            for b in range(inputs.size(0)):
                error_x = out_np[b, 0] - targ_np[b, 0]
                error_y = out_np[b, 1] - targ_np[b, 1]
                euclid_err = np.sqrt(error_x**2 + error_y**2)
                
                original_idx = test_indices[global_idx]
                img_file = full_dataset.labels_df.iloc[original_idx]['image_file']
                
                errors.append({
                    'error': euclid_err,
                    'img_file': img_file,
                    'pred_x': out_np[b, 0],
                    'pred_y': out_np[b, 1],
                    'targ_x': targ_np[b, 0],
                    'targ_y': targ_np[b, 1]
                })
                global_idx += 1
            
    # Sort by highest error
    errors.sort(key=lambda x: x['error'], reverse=True)
    
    worst_16 = errors[:16]
    print(f"Top error: {worst_16[0]['error']:.2f} mm")
    
    # Generate visualization grid
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    fig.suptitle("Top 16 Worst Predictions (Green=Actual, Red=Predicted)", fontsize=16)
    
    for idx, item in enumerate(worst_16):
        ax = axes[idx // 4, idx % 4]
        img_path = os.path.join(images_dir, item['img_file'])
        
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                # Map coordinates to 640x480 for drawing
                def map_to_pixel(x, y):
                    px = int(320 + (x / 140.0) * 640)
                    py = int(240 - (y / 110.0) * 480)
                    return px, py
                    
                tpx, tpy = map_to_pixel(item['targ_x'], item['targ_y'])
                ppx, ppy = map_to_pixel(item['pred_x'], item['pred_y'])
                
                # Draw Actual (Green)
                cv2.circle(img, (tpx, tpy), 15, (0, 255, 0), 3)
                # Draw Predicted (Red)
                cv2.circle(img, (ppx, ppy), 15, (0, 0, 255), 3)
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img)
                ax.set_title(f"Err: {item['error']:.1f}mm\nTarget: ({item['targ_x']:.1f}, {item['targ_y']:.1f})")
        else:
            ax.set_title("Image missing")
            
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved worst predictions grid to: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Find worst predictions")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()
    
    find_worst_predictions(args.model_path, args.data_dir, args.output_path)
