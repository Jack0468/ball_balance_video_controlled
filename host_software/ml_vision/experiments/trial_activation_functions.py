"""Compare ReLU vs. GELU for the Shared Vision Backbone's accuracy, motivated by the
future FPGA/HLS revision (docs/ENGINEERING_STANDARDS.md: floating-point math is
forbidden on-silicon, everything must be fixed-point). ReLU is a single compare-and-
select -- trivial and exact in fixed-point HLS. GELU needs a LUT or polynomial
approximation, which costs FPGA logic and adds an approximation-error verification
surface that ReLU doesn't have. This script doesn't re-litigate that cost difference
(it's well understood); it answers the other half of the decision: does swapping to
ReLU actually cost meaningful accuracy on our data, or is it free?

Production SharedVisionBackbone (host_software/ml_vision/training/train_cnn_2d_tracker_marker.py)
hardcodes GELU at every activation site, so isolating the activation choice requires a
parameterized local copy (TrialSharedVisionBackbone below) rather than a flag on the
real class -- production model is untouched, same pattern trial_augmentation_strategies.py
uses for its dataset class.

Run as a module from the repo root:

    python -m host_software.ml_vision.experiments.trial_activation_functions \
        --subset-size 600 --epochs 8 --num-seeds 3
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from host_software.ml_vision.evaluations.evaluate_shared_vision_backbone import mask_iou_dice
from host_software.ml_vision.experiments.trial_augmentation_strategies import load_subset, split_train_val
from host_software.ml_vision.training.augmentations import build_eval_transform, build_train_transform
from host_software.ml_vision.training.shared_vision_dataset import SharedVisionDataset

DEFAULT_CSV = Path("host_software/data/03_gold/shared_vision/labels.csv")
DEFAULT_IMAGES_DIR = Path("host_software/data/03_gold/shared_vision/images")
DEFAULT_MASKS_DIR = Path("host_software/data/03_gold/shared_vision/masks")
DEFAULT_OUTPUT_DIR = Path("host_software/ml_vision/experiments/results")

INPUT_SIZE = (128, 128)  # (H, W) -- matches production

ACTIVATIONS: Dict[str, Callable[[], nn.Module]] = {
    "relu": lambda: nn.ReLU(inplace=True),
    "gelu": lambda: nn.GELU(),
}


class TrialSharedVisionBackbone(nn.Module):
    """SharedVisionBackbone's architecture with the activation function parameterized.
    Layer shapes are identical to production for every choice of `activation`, so
    parameter count (and therefore capacity) is unaffected by the comparison."""

    def __init__(self, activation: str, input_size: Tuple[int, int] = INPUT_SIZE) -> None:
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"Unknown activation '{activation}'. Choices: {list(ACTIVATIONS)}")
        act = ACTIVATIONS[activation]
        self.input_size = input_size

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            act(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            act(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            act(),
            nn.MaxPool2d(2),
        )

        self.ball_head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            act(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, 2),
        )

        self.mask_head = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.BatchNorm2d(32),
            act(),
            nn.ConvTranspose2d(32, 16, 2, stride=2),
            nn.BatchNorm2d(16),
            act(),
            nn.ConvTranspose2d(16, 8, 2, stride=2),
            nn.BatchNorm2d(8),
            act(),
            nn.Conv2d(8, 1, 1),
        )

        self.heatmap_head = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.BatchNorm2d(32),
            act(),
            nn.ConvTranspose2d(32, 16, 2, stride=2),
            nn.BatchNorm2d(16),
            act(),
            nn.ConvTranspose2d(16, 8, 2, stride=2),
            nn.BatchNorm2d(8),
            act(),
            nn.Conv2d(8, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        ball_xy = self.ball_head(features)

        mask_logits = self.mask_head(features)
        heatmap_logits = self.heatmap_head(features)
        mask_logits = F.interpolate(mask_logits, size=self.input_size, mode="bilinear", align_corners=False)[:, :1, :, :]
        heatmap_logits = F.interpolate(heatmap_logits, size=self.input_size, mode="bilinear", align_corners=False)[:, :1, :, :]
        return ball_xy, mask_logits, heatmap_logits


def train_variant(
    name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    images_dir: Path,
    masks_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    seed: int,
) -> Dict[str, float]:
    torch.manual_seed(seed)

    train_dataset = SharedVisionDataset(
        csv_file="", root_dir=str(images_dir), mask_dir=str(masks_dir),
        input_size=INPUT_SIZE, transform=build_train_transform(INPUT_SIZE), labels_df=train_df,
    )
    val_dataset = SharedVisionDataset(
        csv_file="", root_dir=str(images_dir), mask_dir=str(masks_dir),
        input_size=INPUT_SIZE, transform=build_eval_transform(INPUT_SIZE), labels_df=val_df,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = TrialSharedVisionBackbone(activation=name, input_size=INPUT_SIZE).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    criterion_ball = nn.HuberLoss(delta=2.0)
    criterion_mask = nn.BCEWithLogitsLoss()
    criterion_heatmap = nn.MSELoss()

    px_scale = torch.tensor([INPUT_SIZE[1], INPUT_SIZE[0]], device=device, dtype=torch.float32)
    best = {
        "ball_loss": float("inf"), "mask_loss": float("inf"), "heatmap_mse": float("inf"),
        "total_loss": float("inf"), "ball_px_error": float("inf"), "mask_iou": 0.0, "mask_dice": 0.0,
    }

    for epoch in range(epochs):
        model.train()
        for images, ball_xy, masks, heatmaps in train_loader:
            images, ball_xy, masks, heatmaps = images.to(device), ball_xy.to(device), masks.to(device), heatmaps.to(device)
            optimizer.zero_grad()
            pred_ball_xy, pred_mask_logits, pred_heatmap_logits = model(images)
            loss_ball = criterion_ball(pred_ball_xy, ball_xy)
            loss_mask = criterion_mask(pred_mask_logits, masks)
            loss_heatmap = criterion_heatmap(torch.sigmoid(pred_heatmap_logits), heatmaps)
            loss = loss_ball + 0.5 * loss_mask + 0.1 * loss_heatmap
            loss.backward()
            optimizer.step()

        model.eval()
        val_ball, val_mask, val_heatmap, n_batches = 0.0, 0.0, 0.0, 0
        px_error_sum, iou_sum, dice_sum, n_samples = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for images, ball_xy, masks, heatmaps in val_loader:
                images, ball_xy, masks, heatmaps = images.to(device), ball_xy.to(device), masks.to(device), heatmaps.to(device)
                pred_ball_xy, pred_mask_logits, pred_heatmap_logits = model(images)
                val_ball += criterion_ball(pred_ball_xy, ball_xy).item()
                val_mask += criterion_mask(pred_mask_logits, masks).item()
                val_heatmap += criterion_heatmap(torch.sigmoid(pred_heatmap_logits), heatmaps).item()
                n_batches += 1

                px_error = (pred_ball_xy - ball_xy) * px_scale
                px_error_sum += px_error.norm(dim=1).sum().item()
                batch_iou, batch_dice = mask_iou_dice(pred_mask_logits, masks)
                iou_sum += batch_iou
                dice_sum += batch_dice
                n_samples += images.size(0)

        val_ball /= max(n_batches, 1)
        val_mask /= max(n_batches, 1)
        val_heatmap /= max(n_batches, 1)
        val_total = val_ball + 0.5 * val_mask + 0.1 * val_heatmap
        ball_px_error = px_error_sum / max(n_samples, 1)
        mask_iou = iou_sum / max(n_samples, 1)
        mask_dice = dice_sum / max(n_samples, 1)

        print(
            f"  [{name}] epoch {epoch + 1}/{epochs} | val_total={val_total:.4f} "
            f"ball_px_err={ball_px_error:.2f}px iou={mask_iou:.3f}"
        )

        if val_total < best["total_loss"]:
            best = {
                "ball_loss": val_ball,
                "mask_loss": val_mask,
                "heatmap_mse": val_heatmap,
                "total_loss": val_total,
                "ball_px_error": ball_px_error,
                "mask_iou": mask_iou,
                "mask_dice": mask_dice,
            }

    best["n_params"] = n_params
    return best


def run_sweep(args: argparse.Namespace) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = load_subset(args.csv_file, args.subset_size, args.seed)
    train_df, val_df = split_train_val(df, args.seed)
    print(f"Subset: {len(df)} rows ({len(train_df)} train / {len(val_df)} val)")

    results = []
    for name in args.activations:
        print(f"\nTraining variant: {name}")
        metrics = train_variant(
            name, train_df, val_df, args.images_dir, args.masks_dir,
            args.epochs, args.batch_size, args.lr, device, args.seed,
        )
        results.append({"activation": name, **metrics})

    return pd.DataFrame(results).sort_values("ball_px_error").reset_index(drop=True)


def save_report(results_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = output_dir / f"activation_trial_{timestamp}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved comparison table to {csv_path}")
    print(results_df.to_string(index=False))

    _plot_comparison(
        results_df["activation"], results_df["ball_px_error"], None, results_df["mask_iou"], None,
        output_dir / f"activation_trial_{timestamp}.png", title_suffix="",
    )


def aggregate_seed_runs(runs: List[pd.DataFrame]) -> pd.DataFrame:
    """Average each activation's metrics across seeds -- a single seed's ranking can flip
    on noise alone (see experiments/results/ANALYSIS_2026-08-11.md's Run 1 vs Run 2)."""
    combined = pd.concat(runs, ignore_index=True)
    agg = combined.groupby("activation").agg(
        ball_px_error_mean=("ball_px_error", "mean"),
        ball_px_error_std=("ball_px_error", "std"),
        mask_iou_mean=("mask_iou", "mean"),
        mask_iou_std=("mask_iou", "std"),
        mask_dice_mean=("mask_dice", "mean"),
        heatmap_mse_mean=("heatmap_mse", "mean"),
        total_loss_mean=("total_loss", "mean"),
        total_loss_std=("total_loss", "std"),
        n_params=("n_params", "first"),
        n_seeds=("total_loss", "count"),
    )
    return agg.sort_values("ball_px_error_mean").reset_index()


def save_multi_seed_report(agg_df: pd.DataFrame, raw_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary_path = output_dir / f"activation_trial_multiseed_{timestamp}_summary.csv"
    raw_path = output_dir / f"activation_trial_multiseed_{timestamp}_raw.csv"
    agg_df.to_csv(summary_path, index=False)
    raw_df.to_csv(raw_path, index=False)
    print(f"\nSaved multi-seed summary to {summary_path}")
    print(f"Saved per-seed raw results to {raw_path}")
    print(agg_df.to_string(index=False))

    n_seeds = int(agg_df["n_seeds"].iloc[0])
    _plot_comparison(
        agg_df["activation"], agg_df["ball_px_error_mean"], agg_df["ball_px_error_std"],
        agg_df["mask_iou_mean"], agg_df["mask_iou_std"],
        output_dir / f"activation_trial_multiseed_{timestamp}.png", title_suffix=f" ({n_seeds} seeds)",
    )


def _plot_comparison(labels, px_mean, px_std, iou_mean, iou_std, plot_path: Path, title_suffix: str) -> None:
    fig, (ax_px, ax_iou) = plt.subplots(1, 2, figsize=(11, 5))

    ax_px.bar(labels, px_mean, yerr=(px_std.fillna(0.0) if px_std is not None else None), capsize=4, color="steelblue")
    ax_px.set_ylabel("Val ball position error (px)")
    ax_px.set_title(f"Ball Position Error{title_suffix}")
    ax_px.grid(True, axis="y", alpha=0.3)

    ax_iou.bar(labels, iou_mean, yerr=(iou_std.fillna(0.0) if iou_std is not None else None), capsize=4, color="darkorange")
    ax_iou.set_ylabel("Val mask IoU")
    ax_iou.set_ylim(0, 1)
    ax_iou.set_title(f"Marker Mask IoU{title_suffix}")
    ax_iou.grid(True, axis="y", alpha=0.3)

    fig.suptitle("ReLU vs. GELU -- Shared Vision Backbone")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved comparison chart to {plot_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare ReLU vs. GELU accuracy for the Shared Vision Backbone (FPGA/HLS activation choice)"
    )
    parser.add_argument("--csv-file", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--masks-dir", type=Path, default=DEFAULT_MASKS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--subset-size", type=int, default=600)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-seeds", type=int, default=1,
        help="Run the full sweep this many times (seed, seed+1, ...) and average results, "
        "since a single seed's ranking can flip on noise alone",
    )
    parser.add_argument(
        "--activations", nargs="+", default=list(ACTIVATIONS.keys()), choices=list(ACTIVATIONS.keys()),
        help="Activation functions to compare (default: relu gelu)",
    )
    args = parser.parse_args()

    if not args.csv_file.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv_file}. Run merge_shared_vision_sessions.py first.")

    if args.num_seeds > 1:
        base_seed = args.seed
        runs = []
        for i in range(args.num_seeds):
            args.seed = base_seed + i
            print(f"\n=== Seed {args.seed} ({i + 1}/{args.num_seeds}) ===")
            run_df = run_sweep(args)
            run_df["seed"] = args.seed
            runs.append(run_df)
        agg_df = aggregate_seed_runs(runs)
        save_multi_seed_report(agg_df, pd.concat(runs, ignore_index=True), args.output_dir)
    else:
        results_df = run_sweep(args)
        save_report(results_df, args.output_dir)


if __name__ == "__main__":
    main()
