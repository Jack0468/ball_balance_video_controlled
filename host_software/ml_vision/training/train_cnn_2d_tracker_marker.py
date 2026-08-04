import argparse
import csv
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from host_software.ml_vision.training.shared_vision_dataset import SharedVisionDataset


class SharedVisionBackbone(nn.Module):
    """Small shared backbone with separate regression and segmentation heads."""

    def __init__(self, input_size: Tuple[int, int] = (128, 128)) -> None:
        super().__init__()
        self.input_size = input_size
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
        )

        self.ball_head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, 2),
        )

        # feature_head removed — it had no semantic target and wasted ~5K params

        self.mask_head = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, 2, stride=2),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.ConvTranspose2d(16, 8, 2, stride=2),
            nn.BatchNorm2d(8),
            nn.GELU(),
            nn.Conv2d(8, 1, 1),
        )

        self.heatmap_head = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, 2, stride=2),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.ConvTranspose2d(16, 8, 2, stride=2),
            nn.BatchNorm2d(8),
            nn.GELU(),
            nn.Conv2d(8, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        ball_xy = self.ball_head(features)

        mask_logits = self.mask_head(features)
        heatmap_logits = self.heatmap_head(features)
        mask_logits = F.interpolate(mask_logits, size=self.input_size, mode="bilinear", align_corners=False)
        heatmap_logits = F.interpolate(heatmap_logits, size=self.input_size, mode="bilinear", align_corners=False)
        mask_logits = mask_logits[:, :1, :, :]
        heatmap_logits = heatmap_logits[:, :1, :, :]
        return ball_xy, mask_logits, heatmap_logits


def train_model(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(7)

    if device.type == "cuda":
        cudnn.benchmark = True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.Resize(args.input_size),
            transforms.ToTensor(),
        ]
    )

    dataset = SharedVisionDataset(
        csv_file=args.csv_file,
        root_dir=args.images_dir,
        mask_dir=args.mask_dir,
        input_size=tuple(args.input_size),
        transform=transform,
    )

    if len(dataset) < 2:
        raise ValueError("Dataset must contain at least two samples for train/test splitting")

    train_size = int(0.8 * len(dataset))
    # Shuffle before splitting to prevent chronological train/test leakage
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(42)).tolist()
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = SharedVisionBackbone(input_size=tuple(args.input_size)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=args.patience, factor=0.5)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    criterion_ball = nn.HuberLoss(delta=2.0)
    criterion_mask = nn.BCEWithLogitsLoss()
    criterion_heatmap = nn.MSELoss()  # target is a Gaussian, not the binary mask

    history_train_loss = []
    history_test_loss = []
    best_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for images, ball_xy, masks, heatmap_targets in train_loader:
            images = images.to(device)
            ball_xy = ball_xy.to(device)
            masks = masks.to(device)
            heatmap_targets = heatmap_targets.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                pred_ball_xy, pred_mask_logits, pred_heatmap_logits = model(images)
                loss_ball = criterion_ball(pred_ball_xy, ball_xy)
                loss_mask = criterion_mask(pred_mask_logits, masks)
                # Heatmap trained against Gaussian peaks, not the raw binary mask
                loss_heatmap = criterion_heatmap(torch.sigmoid(pred_heatmap_logits), heatmap_targets)
                loss = loss_ball + 0.5 * loss_mask + 0.1 * loss_heatmap

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_dataset)
        history_train_loss.append(train_loss)

        model.eval()
        running_test_loss = 0.0
        with torch.no_grad():
            for images, ball_xy, masks, heatmap_targets in test_loader:
                images = images.to(device)
                ball_xy = ball_xy.to(device)
                masks = masks.to(device)
                heatmap_targets = heatmap_targets.to(device)
                pred_ball_xy, pred_mask_logits, pred_heatmap_logits = model(images)
                loss_ball = criterion_ball(pred_ball_xy, ball_xy)
                loss_mask = criterion_mask(pred_mask_logits, masks)
                loss_heatmap = criterion_heatmap(torch.sigmoid(pred_heatmap_logits), heatmap_targets)
                test_loss = loss_ball + 0.5 * loss_mask + 0.1 * loss_heatmap
                running_test_loss += test_loss.item() * images.size(0)

        test_loss = running_test_loss / len(test_dataset)
        history_test_loss.append(test_loss)
        scheduler.step(test_loss)

        print(f"Epoch {epoch + 1}/{args.epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

        if test_loss < best_loss:
            best_loss = test_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), output_dir / "shared_vision_backbone_best.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.early_stop_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs (no improvement for {args.early_stop_patience} epochs).")
                break

    torch.save(model.state_dict(), output_dir / "shared_vision_backbone.pt")

    # ONNX export — load the best checkpoint before exporting
    best_ckpt = output_dir / "shared_vision_backbone_best.pt"
    if best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
    model.eval()
    h, w = tuple(args.input_size)
    dummy_input = torch.zeros(1, 3, h, w, device=device)
    onnx_path = output_dir / "shared_vision_backbone_best.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["image"],
        output_names=["ball_xy", "mask_logits", "heatmap_logits"],
        dynamic_axes={"image": {0: "batch"}, "ball_xy": {0: "batch"}, "mask_logits": {0: "batch"}, "heatmap_logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"ONNX model exported to: {onnx_path}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, args.epochs + 1), history_train_loss, label="Train Loss")
    plt.plot(range(1, args.epochs + 1), history_test_loss, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Shared Vision Backbone Training Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "training_curve.png")
    plt.close()

    with (output_dir / "training_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Epoch", "Train_Loss", "Test_Loss"])
        for epoch_idx, (train_loss_value, test_loss_value) in enumerate(zip(history_train_loss, history_test_loss), start=1):
            writer.writerow([epoch_idx, train_loss_value, test_loss_value])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the shared vision backbone")
    parser.add_argument("--csv-file", default="host_software/data/02_silver/labels.csv")
    parser.add_argument("--images-dir", default="host_software/data/02_silver/images")
    parser.add_argument("--mask-dir", default="host_software/data/02_silver/masks")
    parser.add_argument("--output-dir", default="host_software/ml_vision/models/shared_vision_backbone")
    parser.add_argument("--input-size", type=int, nargs=2, default=[128, 128])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=3, help="ReduceLROnPlateau patience")
    parser.add_argument("--early-stop-patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes (0=main thread)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print("Dry run requested; creating a placeholder model file only.")
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"dry_run": True}, output_dir / "shared_vision_backbone.pt")
        return

    train_model(args)


if __name__ == "__main__":
    main()
