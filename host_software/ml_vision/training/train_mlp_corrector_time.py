import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse


class TimeWindowDataset(Dataset):
    def __init__(self, csv_file, window_size=5, future_offset=1, max_gap_ms=50):
        self.window_size = window_size
        self.future_offset = future_offset
        self.max_gap_ms = max_gap_ms

        print(f"Loading {csv_file}...")
        self.df = pd.read_csv(csv_file)

        # Calculate dt (time delta from previous frame in milliseconds)
        # We fill the first row with 33ms (approx 30fps)
        self.df["dt"] = self.df["frame_timestamp_ms"].diff().fillna(33.0)

        self.valid_indices = []

        # We need (window_size) frames for input, plus (future_offset) frames for the target
        required_len = window_size + future_offset

        print("Extracting valid continuous sequences...")
        for i in range(len(self.df) - required_len + 1):
            # Check continuity of the entire sequence block
            block = self.df.iloc[i : i + required_len]
            if len(block) > 1:
                max_gap_in_block = block["dt"].iloc[1:].max()
            else:
                max_gap_in_block = 0.0

            # Check for out-of-bounds (ball fallen off or too close to edge)
            # Platform is 187.5 x 142.0 (centered coordinates mean max is ~93.75 and 71.0)
            max_abs_x = block["touch_x"].abs().max()
            max_abs_y = block["touch_y"].abs().max()

            is_out_of_bounds = max_abs_x > 90.0 or max_abs_y > 68.0

            if max_gap_in_block <= max_gap_ms and not is_out_of_bounds:
                self.valid_indices.append(i)

        print(
            f"Found {len(self.valid_indices)} valid sequences of length {required_len}"
        )

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]

        # Input block (first window_size frames)
        input_block = self.df.iloc[start_idx : start_idx + self.window_size]

        # Features: [cnn_pixel_x, cnn_pixel_y, target_x, target_y, dt]
        features = input_block[
            ["cnn_pixel_x", "cnn_pixel_y", "target_x", "target_y", "dt"]
        ].values.astype(np.float32)

        # Normalize features to roughly [-1, 1]
        features[:, 0] = (
            features[:, 0] / 320.0
        ) - 1.0  # cnn_pixel_x: [0, 640] -> [-1, 1]
        features[:, 1] = (
            features[:, 1] / 240.0
        ) - 1.0  # cnn_pixel_y: [0, 480] -> [-1, 1]
        features[:, 2] = features[:, 2] / 93.75  # target_x: [-93.75, 93.75] -> [-1, 1]
        features[:, 3] = features[:, 3] / 71.0  # target_y: [-71.0, 71.0] -> [-1, 1]
        features[:, 4] = (features[:, 4] / 33.0) - 1.0  # dt: ~33ms -> ~0.0

        # Target frame (at window_size - 1 + future_offset)
        target_frame = self.df.iloc[
            start_idx + self.window_size - 1 + self.future_offset
        ]

        # Targets: [touch_x, touch_y]
        targets = target_frame[["touch_x", "touch_y"]].values.astype(np.float32)

        # Flatten features (e.g., 5 frames * 5 features = 25 size vector)
        return torch.tensor(features.flatten()), torch.tensor(targets)


class MLPCorrectorTime(nn.Module):
    def __init__(self, window_size=5, num_features=5):
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
            nn.Linear(32, 2),  # [touch_x, touch_y]
        )

    def forward(self, x):
        return self.net(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="../../data/02_silver/session_20260730_174916/cnn_sequential_features.csv",
    )
    parser.add_argument(
        "--version", type=str, default="2", help="Model version number (e.g. 1 or v2)"
    )
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument(
        "--future_offset",
        type=int,
        default=0,
        help="0=predict current, 1=predict next frame (~33ms future)",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(args.csv):
        csv_path = os.path.abspath(args.csv)
    else:
        csv_path = os.path.abspath(os.path.join(script_dir, args.csv))

    version_raw = str(args.version)
    version_num = version_raw[1:] if version_raw.startswith("v") else version_raw
    save_dir = os.path.abspath(
        os.path.join(script_dir, f"../models/mlp_corrector_time_v{version_num}")
    )

    os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        print(f"Dataset not found at {csv_path}")
        return

    dataset = TimeWindowDataset(
        csv_path, window_size=args.window_size, future_offset=args.future_offset
    )

    if len(dataset) == 0:
        print("Error: No valid sequences found!")
        return

    # Split 80/20 train/test strictly sequentially
    split_idx = int(len(dataset) * 0.8)
    indices = list(range(len(dataset)))
    train_dataset = Subset(dataset, indices[:split_idx])
    test_dataset = Subset(dataset, indices[split_idx:])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MLPCorrectorTime(window_size=args.window_size).to(device)

    # We use HuberLoss (Smooth L1) for robust regression
    criterion = nn.HuberLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    num_epochs = 50
    best_loss = float("inf")

    train_losses = []
    test_losses = []

    print("Starting Training...")
    for epoch in range(num_epochs):
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

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        val_loss = val_loss / len(test_dataset)
        test_losses.append(val_loss)

        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_path = os.path.join(save_dir, "mlp_corrector_best.pth")
            torch.save(model.state_dict(), best_path)

        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Best: {best_loss:.4f}"
            )

    # Save final model
    torch.save(model.state_dict(), os.path.join(save_dir, "mlp_corrector_final.pth"))

    # Plot curves
    plt.figure()
    plt.plot(train_losses, label="Train")
    plt.plot(test_losses, label="Test")
    plt.title("MLP Time-Series Corrector Loss")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "training_curve.png"))
    print(f"Finished Training! Best Val Loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
