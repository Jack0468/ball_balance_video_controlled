import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from host_software.ml_vision.training.shared_vision_dataset import SharedVisionDataset
from host_software.ml_vision.training.augmentations import build_eval_transform, build_train_transform


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


def temporal_split(
    df: pd.DataFrame, val_fraction: float = 0.2, sort_col: str = "frame_index"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per-session temporal split: within each session, train on the earliest
    (1 - val_fraction) fraction of frames (by sort_col) and validate on the latest
    val_fraction. Consecutive frames are near-duplicates (the ball barely moves
    frame to frame), so a random row-level split leaks near-identical frames
    between train/val and inflates validation accuracy -- see
    docs/PROJECT_LOGBOOK.md (2026-07-13, "Temporal Dataset Restrictions").
    Splitting per-session (rather than on the whole concatenated dataset) also
    guarantees every session contributes a held-out slice, not just whichever
    session happens to land at the tail of the file.
    """
    if "session" not in df.columns:
        raise ValueError(
            "temporal_split requires a 'session' column (produced by merge_shared_vision_sessions.py)"
        )
    if sort_col not in df.columns:
        raise ValueError(f"temporal_split requires a '{sort_col}' column to order frames chronologically")

    train_parts, val_parts = [], []
    for _, group in df.groupby("session"):
        group = group.sort_values(sort_col).reset_index(drop=True)
        split_idx = int(len(group) * (1 - val_fraction))
        split_idx = min(max(split_idx, 1), len(group) - 1) if len(group) > 1 else len(group)
        train_parts.append(group.iloc[:split_idx])
        val_parts.append(group.iloc[split_idx:])

    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True)
    return train_df, val_df


def evaluate_per_session(
    model: SharedVisionBackbone,
    val_df: pd.DataFrame,
    images_dir: str,
    mask_dir: str,
    input_size: Tuple[int, int],
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> pd.DataFrame:
    """Break the validation loss down by session, so a strong aggregate score can't
    hide the model doing badly on one specific sheet (e.g. the blank platform vs. the
    mixed-marker sheets)."""
    eval_transform = build_eval_transform(input_size)
    criterion_ball = nn.HuberLoss(delta=2.0)
    criterion_mask = nn.BCEWithLogitsLoss()
    criterion_heatmap = nn.MSELoss()
    px_scale = torch.tensor([input_size[1], input_size[0]], device=device, dtype=torch.float32)

    model.eval()
    rows = []
    for session_name in sorted(val_df["session"].unique()):
        session_df = val_df[val_df["session"] == session_name].reset_index(drop=True)
        dataset = SharedVisionDataset(
            csv_file="",  # unused -- labels_df takes precedence
            root_dir=images_dir,
            mask_dir=mask_dir,
            input_size=input_size,
            transform=eval_transform,
            labels_df=session_df,
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        ball_sum, mask_sum, heatmap_sum, n_batches = 0.0, 0.0, 0.0, 0
        px_error_sum, n_samples = 0.0, 0
        with torch.no_grad():
            for images, ball_xy, masks, heatmap_targets in loader:
                images, ball_xy, masks, heatmap_targets = (
                    images.to(device), ball_xy.to(device), masks.to(device), heatmap_targets.to(device)
                )
                pred_ball_xy, pred_mask_logits, pred_heatmap_logits = model(images)
                ball_sum += criterion_ball(pred_ball_xy, ball_xy).item()
                mask_sum += criterion_mask(pred_mask_logits, masks).item()
                heatmap_sum += criterion_heatmap(torch.sigmoid(pred_heatmap_logits), heatmap_targets).item()
                n_batches += 1

                px_error = (pred_ball_xy - ball_xy) * px_scale
                px_error_sum += px_error.norm(dim=1).sum().item()
                n_samples += images.size(0)

        ball_loss = ball_sum / max(n_batches, 1)
        mask_loss = mask_sum / max(n_batches, 1)
        heatmap_loss = heatmap_sum / max(n_batches, 1)
        rows.append(
            {
                "session": session_name,
                "n_val_rows": len(session_df),
                "ball_px_error": px_error_sum / max(n_samples, 1),
                "ball_loss": ball_loss,
                "mask_loss": mask_loss,
                "heatmap_loss": heatmap_loss,
                "total_loss": ball_loss + 0.5 * mask_loss + 0.1 * heatmap_loss,
            }
        )

    return pd.DataFrame(rows).sort_values("ball_px_error").reset_index(drop=True)


def _append_training_log_row(output_dir: Path, epoch: int, train_loss: float, test_loss: float, header: bool) -> None:
    """Write one epoch's row immediately rather than buffering the whole log in memory,
    so a mid-training crash doesn't lose the history for epochs that already finished."""
    mode = "w" if header else "a"
    with (output_dir / "training_log.csv").open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if header:
            writer.writerow(["Epoch", "Train_Loss", "Test_Loss"])
        writer.writerow([epoch, train_loss, test_loss])


def _plot_training_curve(history_train_loss: List[float], history_test_loss: List[float], output_dir: Path) -> None:
    """Regenerated every epoch (cheap for a line plot this small) so training_curve.png
    reflects progress even if the run never reaches args.epochs. Sized off len(history)
    rather than args.epochs -- early stopping means those can differ."""
    plt.figure(figsize=(10, 6))
    epochs_range = range(1, len(history_train_loss) + 1)
    plt.plot(epochs_range, history_train_loss, label="Train Loss")
    plt.plot(epochs_range, history_test_loss, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Shared Vision Backbone Training Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "training_curve.png")
    plt.close()


def train_model(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(7)

    if device.type == "cuda":
        cudnn.benchmark = True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_path = output_dir / "shared_vision_backbone_resume.pt"

    labels_df = pd.read_csv(args.csv_file)
    if len(labels_df) < 2:
        raise ValueError("Dataset must contain at least two samples for train/test splitting")

    # Temporal, per-session split -- NOT a random shuffle. Consecutive frames are
    # near-duplicates, so a random split would leak near-identical frames between
    # train/val and inflate validation accuracy. Splitting within each session
    # (train = earliest frames, val = latest frames) also guarantees every session
    # contributes a held-out slice. See temporal_split() docstring for details.
    train_df, test_df = temporal_split(labels_df, val_fraction=args.val_fraction)

    input_size = tuple(args.input_size)
    # OneOf{photometric, shadow, geometric_jitter} -- picked from the augmentation trial
    # (host_software/ml_vision/experiments/, see experiments/results/ANALYSIS_2026-08-11.md).
    # Validation always stays on build_eval_transform (resize-only) so metrics reflect
    # generalisation to clean data, not re-augmented eval noise.
    train_transform = build_eval_transform(input_size) if args.no_augment else build_train_transform(input_size)
    eval_transform = build_eval_transform(input_size)

    train_dataset = SharedVisionDataset(
        csv_file=args.csv_file,
        root_dir=args.images_dir,
        mask_dir=args.mask_dir,
        input_size=input_size,
        transform=train_transform,
        labels_df=train_df,
    )
    test_dataset = SharedVisionDataset(
        csv_file=args.csv_file,
        root_dir=args.images_dir,
        mask_dir=args.mask_dir,
        input_size=input_size,
        transform=eval_transform,
        labels_df=test_df,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = SharedVisionBackbone(input_size=tuple(args.input_size)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=args.patience, factor=0.5)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    criterion_ball = nn.HuberLoss(delta=2.0)
    criterion_mask = nn.BCEWithLogitsLoss()
    criterion_heatmap = nn.MSELoss()  # target is a Gaussian, not the binary mask

    start_epoch = 0
    history_train_loss: List[float] = []
    history_test_loss: List[float] = []
    best_loss = float("inf")
    epochs_no_improve = 0

    # --resume picks this back up after a Colab disconnect/crash -- reloads model,
    # optimizer, and scheduler state (not just weights, so AdamW's momentum/variance
    # buffers and the LR schedule don't restart cold) plus the loss history, and
    # continues from the next epoch instead of retraining from scratch. Safe to always
    # pass --resume: if no checkpoint exists yet, this just falls through and trains fresh.
    if args.resume and resume_path.exists():
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["best_loss"]
        epochs_no_improve = checkpoint["epochs_no_improve"]
        history_train_loss = checkpoint["history_train_loss"]
        history_test_loss = checkpoint["history_test_loss"]
        print(f"Resumed from {resume_path} at epoch {start_epoch} (best_loss={best_loss:.4f})")
    elif args.resume:
        print(f"--resume was set but no checkpoint found at {resume_path}; training from scratch.")

    if start_epoch >= args.epochs:
        print(f"Resume checkpoint is already at epoch {start_epoch} >= --epochs {args.epochs}; nothing to train.")

    for epoch in range(start_epoch, args.epochs):
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

        # Per-epoch log row + curve refresh + full training-state checkpoint, all written
        # immediately -- so a mid-training Colab crash loses at most the in-progress epoch,
        # not the whole run. shared_vision_backbone_resume.pt carries optimizer/scheduler
        # state and history (for --resume); shared_vision_backbone_best.pt above stays a
        # plain model state_dict so evaluate_shared_vision_backbone.py / ONNX export don't
        # need to know about the resume format.
        _append_training_log_row(
            output_dir, epoch + 1, train_loss, test_loss, header=(epoch == 0 and start_epoch == 0)
        )
        _plot_training_curve(history_train_loss, history_test_loss, output_dir)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_loss": best_loss,
                "epochs_no_improve": epochs_no_improve,
                "history_train_loss": history_train_loss,
                "history_test_loss": history_test_loss,
            },
            resume_path,
        )

        if epochs_no_improve >= args.early_stop_patience:
            print(f"Early stopping triggered after {epoch + 1} epochs (no improvement for {args.early_stop_patience} epochs).")
            break

    torch.save(model.state_dict(), output_dir / "shared_vision_backbone.pt")

    # ONNX export — load the best checkpoint before exporting
    best_ckpt = output_dir / "shared_vision_backbone_best.pt"
    if best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
    model.eval()

    # Per-session breakdown on the held-out temporal slice -- a strong aggregate score
    # can hide the model doing badly on one specific sheet.
    per_session_df = evaluate_per_session(
        model, test_df, args.images_dir, args.mask_dir, input_size, args.batch_size, args.num_workers, device
    )
    per_session_path = output_dir / "per_session_eval.csv"
    per_session_df.to_csv(per_session_path, index=False)
    print(f"\nPer-session validation performance (saved to {per_session_path}):")
    print(per_session_df.to_string(index=False))

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
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of each session's frames (by frame_index, chronologically latest) held out for validation",
    )
    parser.add_argument("--patience", type=int, default=3, help="ReduceLROnPlateau patience")
    parser.add_argument("--early-stop-patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes (0=main thread)")
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable training-time augmentation (OneOf{photometric, shadow, geometric_jitter}); "
        "use resize-only, same as validation. Every augmented variant beat baseline in the "
        "trial (see experiments/results/ANALYSIS_2026-08-11.md), so augmentation is on by default.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from output-dir/shared_vision_backbone_resume.pt if present. Safe to "
        "always pass -- trains from scratch if no checkpoint is found there yet.",
    )
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
