import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import argparse
from unified_dataset import UnifiedDataset
from basic_cnn import BasicCNN


def unified_loss(preds, targets):
    """
    Computes a combined loss for the 33-output Unified CNN.
    preds: (Batch, 33)
    targets: (Batch, 33)
    """
    # 0:32 are coordinates, 32 is ball presence
    pred_coords = preds[:, :32]
    target_coords = targets[:, :32]

    pred_presence = preds[:, 32]
    target_presence = targets[:, 32]

    # 1. Coordinate Loss (MSE)
    # We only compute loss for targets that are >= 0.0 (ignoring -1.0 missing markers)
    mask = target_coords >= 0.0

    if mask.sum() > 0:
        mse_loss_fn = nn.MSELoss(reduction="mean")
        loss_coords = mse_loss_fn(pred_coords[mask], target_coords[mask])
    else:
        loss_coords = torch.tensor(0.0, device=preds.device)

    # 2. Presence Loss (BCE With Logits)
    bce_loss_fn = nn.BCEWithLogitsLoss()
    loss_presence = bce_loss_fn(pred_presence, target_presence)

    # Combined Loss
    return loss_coords + (
        0.1 * loss_presence
    )  # Weight presence slightly less to prioritize accuracy of coordinates


def main():
    parser = argparse.ArgumentParser(description="Train Unified CNN Tracker")
    parser.add_argument(
        "--data_dir",
        default="../../data/02_silver/session_20260728_102908",
        help="Path to session data directory",
    )
    parser.add_argument(
        "--save_dir", default="../models", help="Directory to save the trained models"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint (.pth) to resume training from",
    )
    args = parser.parse_args()

    print("Initializing PyTorch Unified CNN Tracker (33 outputs)...")
    # Output size is 33 (8 kpts + 2 ball + 22 markers + 1 presence)
    model = BasicCNN(num_outputs=33)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    model = model.to(device)

    data_dir = os.path.abspath(args.data_dir)
    print(f"Loading unified dataset from: {data_dir}")

    train_dataset = UnifiedDataset(data_dir, split="train")
    test_dataset = UnifiedDataset(data_dir, split="test")

    print(f"Train samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    start_epoch = 0
    best_loss = float("inf")
    project_dir = os.path.abspath(args.save_dir)
    os.makedirs(os.path.join(project_dir, "cnn_unified_tracker_v1"), exist_ok=True)

    if args.resume and os.path.isfile(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        if "best_loss" in checkpoint:
            best_loss = checkpoint["best_loss"]
        print(f"Resumed at epoch {start_epoch} with best_loss {best_loss:.4f}")

    num_epochs = 50
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    import csv

    log_path = os.path.join(project_dir, "cnn_unified_tracker_v1/training_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "test_loss"])

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)

            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    outputs = model(inputs)
                    loss = unified_loss(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = unified_loss(outputs, targets)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_train_loss = running_loss / len(train_dataset)

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                if scaler is not None:
                    with torch.amp.autocast("cuda"):
                        outputs = model(inputs)
                        loss = unified_loss(outputs, targets)
                else:
                    outputs = model(inputs)
                    loss = unified_loss(outputs, targets)
                test_loss += loss.item() * inputs.size(0)

        epoch_test_loss = test_loss / len(test_dataset)

        print(
            f"--- Epoch [{epoch+1}/{num_epochs}] Train Loss: {epoch_train_loss:.4f} | Test Loss: {epoch_test_loss:.4f} ---"
        )

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, epoch_train_loss, epoch_test_loss])

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_loss": best_loss,
        }

        save_path = os.path.join(
            project_dir, "cnn_unified_tracker_v1/unified_tracker_best.pth"
        )
        if epoch_test_loss < best_loss:
            best_loss = epoch_test_loss
            torch.save(checkpoint, save_path)
            print(f"Saved new best model to {save_path}")

        latest_path = os.path.join(
            project_dir, "cnn_unified_tracker_v1/unified_tracker_latest.pth"
        )
        torch.save(checkpoint, latest_path)

    print("Training complete!")


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()
