import os
import argparse
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
import time

# Ensure training module is in path to import BallDataset
script_dir = os.path.dirname(os.path.abspath(__file__))
training_dir = os.path.abspath(os.path.join(script_dir, '../training'))
if training_dir not in sys.path:
    sys.path.append(training_dir)

from ball_dataset import BallDataset

def main():
    parser = argparse.ArgumentParser(description="Quick Visual Evaluation for Partially Trained Model")
    parser.add_argument("--data_dir", default="../data/02_silver", help="Path to data directory")
    parser.add_argument("--model_path", default="../models/resnet18_expert_tracker/expert_tracker_latest.pth", help="Path to the model checkpoint")
    parser.add_argument("--num_images", type=int, default=6, help="Number of random images to visualize")
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
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
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
    
    # Pick random indices
    indices = random.sample(range(len(full_dataset)), args.num_images)
    subset = Subset(full_dataset, indices)
    loader = DataLoader(subset, batch_size=args.num_images, shuffle=False)
    
    MAX_BOUND = 200.0 # From ball_dataset.py

    print("Running inference...")
    inputs, targets = next(iter(loader))
    inputs = inputs.to(device)
    
    with torch.no_grad():
        t0 = time.perf_counter()
        outputs = model(inputs)
        t1 = time.perf_counter()
        
    print(f"Inference completed in {(t1-t0)*1000.0:.2f} ms for {args.num_images} images.")

    # Convert to numpy and de-normalize coordinates
    preds_mm = outputs.cpu().numpy() * MAX_BOUND
    targs_mm = targets.cpu().numpy() * MAX_BOUND
    
    # Denormalize images for plotting
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Quick Visual Evaluation (Partially Trained Model)\nGreen = Ground Truth | Red = Prediction', fontsize=16)
    
    axes = axes.flatten()
    
    for i in range(min(args.num_images, len(axes))):
        img = inputs[i].cpu()
        img = inv_normalize(img)
        img = img.numpy().transpose(1, 2, 0)
        img = np.clip(img, 0, 1)
        
        # Ground truth
        t_x_mm, t_y_mm = targs_mm[i]
        t_px_x = t_x_mm + 160.0 # Center offset
        t_px_y = 120.0 - t_y_mm # Y is usually inverted in images
        
        # Prediction
        p_x_mm, p_y_mm = preds_mm[i]
        p_px_x = p_x_mm + 160.0
        p_px_y = 120.0 - p_y_mm

        ax = axes[i]
        ax.imshow(img)
        
        # Plot target
        ax.plot(t_px_x, t_px_y, 'g+', markersize=15, markeredgewidth=2, label='Actual')
        # Plot prediction
        ax.plot(p_px_x, p_px_y, 'rx', markersize=15, markeredgewidth=2, label='Pred')
        
        ax.set_title(f"Target: ({t_x_mm:.1f}, {t_y_mm:.1f})\nPred: ({p_x_mm:.1f}, {p_y_mm:.1f})")
        ax.axis('off')
        
    handles, labels = ax.get_legend_handles_labels()
    # Remove duplicates from legend
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='lower center', ncol=2, fontsize=12)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    output_path = os.path.join(script_dir, 'quick_eval_visual.png')
    plt.savefig(output_path)
    print(f"\nSaved visualization to {output_path}")

if __name__ == '__main__':
    main()
