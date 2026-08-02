import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))


class UnifiedCorrectorMLP(nn.Module):
    """
    Takes 32 pixel coordinates (Platform corners, Ball, Markers) and outputs the physical
    (touch_x, touch_y) in millimeters.
    This bypasses the need for explicit Homography calculations.
    """

    def __init__(self, input_dim=32, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, x):
        return self.net(x)


class UnifiedYoloFeatureDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

        # 32 features: 8 kpts + 2 ball + 22 markers
        feature_cols = [
            "kpt0_x",
            "kpt0_y",
            "kpt1_x",
            "kpt1_y",
            "kpt2_x",
            "kpt2_y",
            "kpt3_x",
            "kpt3_y",
            "ball_x",
            "ball_y",
        ]
        for c in range(2, 13):
            feature_cols.extend([f"marker{c}_x", f"marker{c}_y"])

        self.X = self.df[feature_cols].values.astype("float32")
        self.y = self.df[["touch_x", "touch_y"]].values.astype("float32")

        # Normalize the inputs to help the MLP learn faster [0, 1]
        # Assume max image dimensions are 640x480 (from YOLO)
        self.X[:, 0::2] = np.where(
            self.X[:, 0::2] >= 0, self.X[:, 0::2] / 640.0, -1.0
        )  # x coords
        self.X[:, 1::2] = np.where(
            self.X[:, 1::2] >= 0, self.X[:, 1::2] / 480.0, -1.0
        )  # y coords

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])


import numpy as np


def train():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_csv",
        default="../../data/02_silver/session_20260728_102908/yolo_features.csv",
        help="Path to yolo_features.csv",
    )
    parser.add_argument(
        "--epochs", type=int, default=300, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    csv_path = os.path.abspath(os.path.join(script_dir, args.data_csv))
    if not os.path.exists(csv_path):
        print(f"ERROR: Could not find {csv_path}")
        return

    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Filter out missing ball detections since corrector needs the ball to predict touch_x
    df = df[df["ball_present"] == 1.0].reset_index(drop=True)

    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    split_idx = int(0.8 * len(df))
    train_dataset = UnifiedYoloFeatureDataset(df.iloc[:split_idx])
    test_dataset = UnifiedYoloFeatureDataset(df.iloc[split_idx:])

    print(
        f"Loaded {len(train_dataset)} training and {len(test_dataset)} testing samples."
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UnifiedCorrectorMLP().to(device)
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    train_losses = []
    test_losses = []

    best_test_loss = float("inf")
    save_dir = os.path.abspath(
        os.path.join(script_dir, "../models/unified_corrector_v1")
    )
    os.makedirs(save_dir, exist_ok=True)
    model_save_path = os.path.join(save_dir, "best_unified_corrector.pth")

    print("Starting training Unified Corrector...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            # Jitter Augmentation: Add small Gaussian noise to valid coordinates
            noise = torch.randn_like(X) * 0.005
            # Don't add noise to missing features (-1.0)
            noise = torch.where(X >= 0, noise, torch.zeros_like(noise))
            X_noisy = X + noise

            optimizer.zero_grad()
            out = model(X_noisy)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                loss = criterion(out, y)
                total_test_loss += loss.item()

        avg_test_loss = total_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(model.state_dict(), model_save_path)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch+1}/{args.epochs}] Train Loss (Huber): {avg_train_loss:.2f} | Test Loss: {avg_test_loss:.2f}"
            )

    print(f"Training complete! Best model saved to {model_save_path}")

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Huber)")
    plt.legend()
    plt.title("Unified Corrector MLP Training Curve")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "training_curve.png"))
    print("Saved training_curve.png")


if __name__ == "__main__":
    train()
