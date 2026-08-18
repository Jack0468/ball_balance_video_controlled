"""MLP time-corrector for shared_vision_backbone_v2's raw predictions.

Adapted from train_mlp_corrector_time.py -- NOT a blind reuse. That script's
TimeWindowDataset hardcodes /320.0, /240.0 normalization tied to the old
model's raw full-camera-frame pixel output. shared_vision_backbone_v2's
predictions are already in centered platform mm (see
run_shared_vision_inference_on_dataset.py), so the correct normalization here
divides by the platform half-extents (93.75mm / 71.0mm), matching how
target_x/target_y were already normalized in the old script.

Also, unlike the old script's single-session cnn_sequential_features.csv, the
input here (shared_vision_v2_inference_predictions.csv) concatenates 4
sessions -- window construction must not cross session boundaries, and the
held-out split must be per-session (temporal_split), not one 80/20 global
sequential cut, matching the rest of this project's evaluation methodology.

Run as a module from the repo root:

    python -m host_software.ml_vision.training.train_mlp_corrector_shared_vision \
        --csv host_software/ml_vision/evaluations/reports/shared_vision_v2_inference_predictions.csv \
        --version 1
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from host_software.ml_vision.training.train_cnn_2d_tracker_marker import temporal_split

TOUCHPAD_HALF_W_MM = 93.75
TOUCHPAD_HALF_H_MM = 71.0


class TimeWindowDatasetSharedVision(Dataset):
    """Same windowing idea as TimeWindowDataset (train_mlp_corrector_time.py),
    generalized to a per-session-grouped, multi-session dataframe."""

    def __init__(self, df: pd.DataFrame, window_size: int = 5, future_offset: int = 0, max_gap_ms: float = 50.0):
        self.window_size = window_size
        self.future_offset = future_offset
        self.max_gap_ms = max_gap_ms

        df = df.sort_values(["session", "frame_index"]).reset_index(drop=True)
        df["dt"] = df.groupby("session")["frame_timestamp_ms"].diff().fillna(33.0)
        self.df = df

        self.valid_indices = []
        required_len = window_size + future_offset

        for _, group in df.groupby("session"):
            start = group.index[0]
            end = group.index[-1]
            for i in range(start, end - required_len + 2):
                block = df.iloc[i : i + required_len]
                # Never cross a session boundary.
                if block["session"].nunique() > 1:
                    continue
                max_gap_in_block = block["dt"].iloc[1:].max() if len(block) > 1 else 0.0
                max_abs_x = block["touch_x"].abs().max()
                max_abs_y = block["touch_y"].abs().max()
                is_out_of_bounds = max_abs_x > 90.0 or max_abs_y > 68.0
                if max_gap_in_block <= max_gap_ms and not is_out_of_bounds:
                    self.valid_indices.append(i)

        print(f"Found {len(self.valid_indices)} valid sequences of length {required_len}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]
        input_block = self.df.iloc[start_idx : start_idx + self.window_size]

        # Features: [pred_x, pred_y, target_x, target_y, dt] -- pred_x/y already in
        # centered mm (not raw camera pixels), so normalize by platform half-extent,
        # not /320.0 / /240.0.
        features = input_block[["pred_x", "pred_y", "target_x", "target_y", "dt"]].values.astype(np.float32)
        features[:, 0] = features[:, 0] / TOUCHPAD_HALF_W_MM
        features[:, 1] = features[:, 1] / TOUCHPAD_HALF_H_MM
        features[:, 2] = features[:, 2] / TOUCHPAD_HALF_W_MM
        features[:, 3] = features[:, 3] / TOUCHPAD_HALF_H_MM
        features[:, 4] = (features[:, 4] / 33.0) - 1.0

        target_frame = self.df.iloc[start_idx + self.window_size - 1 + self.future_offset]
        targets = target_frame[["touch_x", "touch_y"]].values.astype(np.float32)

        return torch.tensor(features.flatten()), torch.tensor(targets)


class MLPCorrectorSharedVision(nn.Module):
    def __init__(self, window_size: int = 5, num_features: int = 5):
        super().__init__()
        input_dim = window_size * num_features
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.net(x)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MLP time corrector for shared_vision_backbone_v2")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--version", type=str, default="1")
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--future-offset", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    version_num = args.version[1:] if args.version.startswith("v") else args.version
    save_dir = os.path.abspath(os.path.join(script_dir, f"../models/mlp_corrector_shared_vision_v{version_num}"))
    os.makedirs(save_dir, exist_ok=True)

    df = pd.read_csv(args.csv)
    train_df, val_df = temporal_split(df, val_fraction=args.val_fraction, sort_col="frame_index")

    train_dataset = TimeWindowDatasetSharedVision(train_df, window_size=args.window_size, future_offset=args.future_offset)
    val_dataset = TimeWindowDatasetSharedVision(val_df, window_size=args.window_size, future_offset=args.future_offset)

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("Error: No valid sequences found in train or val split!")
        return

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MLPCorrectorSharedVision(window_size=args.window_size).to(device)

    criterion = nn.HuberLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

    best_loss = float("inf")
    train_losses, val_losses = [], []

    print(f"Starting training ({len(train_dataset)} train / {len(val_dataset)} val sequences)...")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        train_loss = running_loss / len(train_dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        val_loss = val_loss / len(val_dataset)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "mlp_corrector_best.pth"))

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Best: {best_loss:.4f}")

    torch.save(model.state_dict(), os.path.join(save_dir, "mlp_corrector_final.pth"))

    plt.figure()
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.title("MLP Corrector (shared_vision_backbone_v2) Loss")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "training_curve.png"))
    print(f"Finished training! Best val loss: {best_loss:.4f}. Saved to {save_dir}")


if __name__ == "__main__":
    main()
