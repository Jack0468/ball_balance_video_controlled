import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
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
training_dir = os.path.abspath(os.path.join(script_dir, "../training"))
if training_dir not in sys.path:
    sys.path.append(training_dir)

import argparse
from ball_dataset import BallDataset


class SpatialSoftmaxResNetHead(nn.Module):
    def __init__(self, in_channels, height, width):
        super(SpatialSoftmaxResNetHead, self).__init__()
        self.height = height
        self.width = width
        self.in_channels = in_channels

        # 1x1 conv to squash the channels down into a single heatmap
        self.heatmap_conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        # Initialize with near-zero variance for a flat starting softmax distribution
        nn.init.normal_(self.heatmap_conv.weight, mean=0, std=0.01)
        if self.heatmap_conv.bias is not None:
            nn.init.constant_(self.heatmap_conv.bias, 0)

    def forward(self, x):
        # x is the flattened output from ResNet (Batch, in_channels * H * W)
        B = x.size(0)
        x = x.view(B, self.in_channels, self.height, self.width)

        heatmap = self.heatmap_conv(x)  # (Batch, 1, H, W)

        # Flatten spatial dims to apply softmax
        heatmap_flat = heatmap.view(B, 1, -1)
        attention = torch.nn.functional.softmax(heatmap_flat, dim=-1)
        attention = attention.view(B, 1, self.height, self.width)

        # Create dynamic grid on the same device as the input tensor
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, self.height, device=x.device),
            torch.linspace(-1, 1, self.width, device=x.device),
            indexing="ij",
        )

        expected_x = torch.sum(attention * grid_x, dim=(2, 3))
        expected_y = torch.sum(attention * grid_y, dim=(2, 3))

        return torch.cat([expected_x, expected_y], dim=1)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Spatial ResNet Expert Tracker"
    )
    parser.add_argument(
        "--data_dir",
        default="../../data/02_silver/session_20260728_102908",
        help="Path to data directory",
    )
    parser.add_argument(
        "--csv_name",
        type=str,
        default="labels_sequential.csv",
        help="Name of the CSV labels file",
    )
    parser.add_argument(
        "--model_path", required=True, help="Path to the trained .pth file"
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet50"],
        help="Architecture to use",
    )
    args = parser.parse_args()

    print(
        f"Initializing Temporary Evaluation Script for Spatial Expert Tracker ({args.arch})..."
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    script_dir = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.abspath(args.model_path)

    project_dir = os.path.dirname(model_path)

    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}. Train it first.")
        return

    # 1. Initialize model
    if args.arch == "resnet18":
        model = models.resnet18(weights=None)
        img_size = (240, 320)
        feature_map_h, feature_map_w = 8, 10
    elif args.arch == "resnet50":
        model = models.resnet50(weights=None)
        img_size = (480, 640)
        feature_map_h, feature_map_w = 15, 20

    num_ftrs = model.fc.in_features
    # We MUST bypass the average pooling to preserve spatial dimensions
    model.avgpool = nn.Identity()
    # Use Spatial Softmax head instead of Linear
    model.fc = SpatialSoftmaxResNetHead(
        in_channels=num_ftrs, height=feature_map_h, width=feature_map_w
    )

    # Load weights safely
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    # 2. Load the test subset data
    data_dir = os.path.abspath(args.data_dir)

    csv_path = os.path.join(data_dir, args.csv_name)
    if not os.path.exists(csv_path):
        csv_path = os.path.join(data_dir, "labels.csv")

    images_dir = os.path.join(data_dir, "images")

    print(f"Loading dataset from: {csv_path}")
    test_transform = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    full_dataset = BallDataset(
        csv_file=csv_path, root_dir=images_dir, transform=test_transform
    )

    # We want to test on the last 20% of the dataset
    indices = list(range(len(full_dataset)))
    train_size = int(0.8 * len(indices))
    test_indices = indices[train_size:]

    test_dataset = Subset(full_dataset, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    print(f"Evaluating on {len(test_dataset)} unseen test frames.")

    MAX_X_BOUND, MAX_Y_BOUND = 200.0, 200.0

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

            # De-normalize
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
        "FPS_Estimate": float(1000.0 / np.mean(inference_times_ms)),
    }

    print("\n--- Evaluation Metrics (Millimeters) ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.2f} mm")

    metrics_path = os.path.join(project_dir, "evaluation_metrics_spatial.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    # Plotting X and Y trajectories
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(targs_x, label="Actual X", color="blue", alpha=0.7)
    plt.plot(preds_x, label="Predicted X", color="red", linestyle="--", alpha=0.7)
    plt.ylabel("X Position (mm)")
    plt.title("Contiguous Test Set Trajectory (X)")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(targs_y, label="Actual Y", color="blue", alpha=0.7)
    plt.plot(preds_y, label="Predicted Y", color="red", linestyle="--", alpha=0.7)
    plt.xlabel("Frame Index (Time)")
    plt.ylabel("Y Position (mm)")
    plt.title("Contiguous Test Set Trajectory (Y)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(project_dir, "evaluation_trajectory_spatial.png")
    plt.savefig(plot_path)
    print(f"\nSaved trajectory plot to {plot_path}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
