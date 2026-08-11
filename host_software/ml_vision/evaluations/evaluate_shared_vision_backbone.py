"""Standalone evaluation for the Shared Vision Backbone (ball + marker heads).

train_cnn_2d_tracker_marker.py already writes a per-session loss breakdown
(per_session_eval.csv) and a loss curve during training, but doesn't compute
segmentation quality (IoU/Dice), inference latency, or any qualitative
prediction visuals. This script loads a trained checkpoint, re-derives the
same held-out temporal validation split used at training time, and reports
those metrics plus a qualitative grid of predictions vs. ground truth.

Run as a module from the repo root (same reason train_cnn_2d_tracker_marker.py
is invoked with -m: it imports host_software.ml_vision.training.* as
package-qualified paths):

    python -m host_software.ml_vision.evaluations.evaluate_shared_vision_backbone \
        --csv-file host_software/data/03_gold/shared_vision/labels.csv \
        --images-dir host_software/data/03_gold/shared_vision/images \
        --mask-dir host_software/data/03_gold/shared_vision/masks \
        --checkpoint /content/drive/MyDrive/VRI_Models/shared_vision_backbone_v1/shared_vision_backbone_best.pt
"""

import argparse
import json
import time
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from host_software.ml_vision.training.augmentations import build_eval_transform
from host_software.ml_vision.training.shared_vision_dataset import SharedVisionDataset
from host_software.ml_vision.training.train_cnn_2d_tracker_marker import (
    SharedVisionBackbone,
    temporal_split,
)


def mask_iou_dice(pred_logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> Tuple[float, float]:
    """Summed IoU and Dice over a batch of binary masks, thresholded at `threshold`."""
    pred = (torch.sigmoid(pred_logits) > threshold).float()
    target = (target > threshold).float()

    dims = (1, 2, 3)
    intersection = (pred * target).sum(dim=dims)
    union = pred.sum(dim=dims) + target.sum(dim=dims) - intersection
    dice_denom = pred.sum(dim=dims) + target.sum(dim=dims)

    iou = torch.where(union > 0, intersection / union, torch.ones_like(union))
    dice = torch.where(dice_denom > 0, 2 * intersection / dice_denom, torch.ones_like(dice_denom))
    return iou.sum().item(), dice.sum().item()


def evaluate(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = tuple(args.input_size)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Train it first.")
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    model = SharedVisionBackbone(input_size=input_size).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Re-derive the same held-out temporal slice used during training (same
    # val_fraction, same per-session split logic) so this evaluates on frames
    # the model never trained on -- see temporal_split() in the training script.
    labels_df = pd.read_csv(args.csv_file)
    _, val_df = temporal_split(labels_df, val_fraction=args.val_fraction)

    dataset = SharedVisionDataset(
        csv_file="",
        root_dir=args.images_dir,
        mask_dir=args.mask_dir,
        input_size=input_size,
        transform=build_eval_transform(input_size),
        labels_df=val_df,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f"Evaluating on {len(dataset)} held-out validation frames.")

    px_scale = torch.tensor([input_size[1], input_size[0]], device=device, dtype=torch.float32)
    heatmap_criterion = torch.nn.MSELoss(reduction="sum")

    px_errors = []
    iou_sum, dice_sum, n_masks = 0.0, 0.0, 0
    heatmap_sq_error_sum, n_heatmap_px = 0.0, 0
    inference_times_ms = []

    with torch.no_grad():
        for images, ball_xy, masks, heatmap_targets in loader:
            images, ball_xy, masks, heatmap_targets = (
                images.to(device), ball_xy.to(device), masks.to(device), heatmap_targets.to(device)
            )

            t0 = time.perf_counter()
            pred_ball_xy, pred_mask_logits, pred_heatmap_logits = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            per_frame_ms = ((t1 - t0) / images.size(0)) * 1000.0
            inference_times_ms.extend([per_frame_ms] * images.size(0))

            px_error = (pred_ball_xy - ball_xy) * px_scale
            px_errors.extend(px_error.norm(dim=1).cpu().numpy().tolist())

            batch_iou, batch_dice = mask_iou_dice(pred_mask_logits, masks)
            iou_sum += batch_iou
            dice_sum += batch_dice
            n_masks += images.size(0)

            heatmap_sq_error_sum += heatmap_criterion(torch.sigmoid(pred_heatmap_logits), heatmap_targets).item()
            n_heatmap_px += heatmap_targets.numel()

    px_errors = np.array(px_errors)
    inference_times_ms = np.array(inference_times_ms)

    metrics = {
        "num_val_samples": int(len(dataset)),
        "ball_px_error_mean": float(px_errors.mean()),
        "ball_px_error_median": float(np.median(px_errors)),
        "ball_px_error_p95": float(np.percentile(px_errors, 95)),
        "ball_px_error_max": float(px_errors.max()),
        "mask_iou_mean": float(iou_sum / max(n_masks, 1)),
        "mask_dice_mean": float(dice_sum / max(n_masks, 1)),
        "heatmap_mse": float(heatmap_sq_error_sum / max(n_heatmap_px, 1)),
        "mean_inference_time_ms": float(inference_times_ms.mean()),
        "p95_inference_time_ms": float(np.percentile(inference_times_ms, 95)),
        "fps_estimate": float(1000.0 / inference_times_ms.mean()),
    }

    print("\n--- Shared Vision Backbone Evaluation Metrics ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.3f}")

    if metrics["mask_iou_mean"] < 0.3:
        print("\n[WARNING] Mean mask IoU is low (<0.3) -- segmentation head may not be learning markers well.")

    metrics_path = output_dir / "evaluation_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"\nSaved metrics to {metrics_path}")

    # Ball pixel error distribution -- complements the per-session breakdown
    # already written by train_cnn_2d_tracker_marker.py's evaluate_per_session().
    plt.figure(figsize=(8, 5))
    plt.hist(px_errors, bins=30, color="steelblue", edgecolor="black")
    plt.axvline(metrics["ball_px_error_mean"], color="red", linestyle="--", label="mean")
    plt.axvline(metrics["ball_px_error_p95"], color="orange", linestyle="--", label="p95")
    plt.xlabel("Ball position error (px)")
    plt.ylabel("Count")
    plt.title("Held-out Ball Position Error Distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    hist_path = output_dir / "evaluation_error_histogram.png"
    plt.savefig(hist_path)
    plt.close()
    print(f"Saved error histogram to {hist_path}")

    _save_qualitative_grid(model, dataset, input_size, device, output_dir, args.num_vis_samples)


def _save_qualitative_grid(
    model: SharedVisionBackbone,
    dataset: SharedVisionDataset,
    input_size: Tuple[int, int],
    device: torch.device,
    output_dir: Path,
    num_samples: int,
) -> None:
    """Grid of held-out frames: predicted vs. ground-truth ball point, and
    predicted mask contour overlaid on the ground-truth mask, for a quick
    visual sanity check that the aggregate metrics above don't capture."""
    num_samples = min(num_samples, len(dataset))
    if num_samples <= 0:
        return
    indices = np.linspace(0, len(dataset) - 1, num_samples, dtype=int)
    h, w = input_size

    cols = min(4, num_samples)
    rows = int(np.ceil(num_samples / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_1d(axes).flatten()

    model.eval()
    with torch.no_grad():
        for ax, idx in zip(axes, indices):
            image, ball_xy, mask, _ = dataset[idx]
            pred_ball_xy, pred_mask_logits, _ = model(image.unsqueeze(0).to(device))
            pred_mask = (torch.sigmoid(pred_mask_logits[0, 0]) > 0.5).cpu().numpy()

            img_np = image.permute(1, 2, 0).numpy()
            ax.imshow(img_np)
            ax.contour(mask[0].numpy(), levels=[0.5], colors="lime", linewidths=1.5)
            ax.contour(pred_mask, levels=[0.5], colors="red", linewidths=1.5)

            gt_x, gt_y = ball_xy[0].item() * w, ball_xy[1].item() * h
            pred_x, pred_y = pred_ball_xy[0, 0].item() * w, pred_ball_xy[0, 1].item() * h
            ax.scatter([gt_x], [gt_y], c="lime", marker="x", s=80, label="GT ball")
            ax.scatter([pred_x], [pred_y], c="red", marker="x", s=80, label="Pred ball")
            ax.set_title(f"idx {idx}", fontsize=9)
            ax.axis("off")

    for ax in axes[num_samples:]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4)
    fig.suptitle("Held-out Predictions (green=ground truth, red=predicted)")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))

    grid_path = output_dir / "evaluation_visual_grid.png"
    fig.savefig(grid_path)
    plt.close(fig)
    print(f"Saved qualitative prediction grid to {grid_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the shared vision backbone")
    parser.add_argument("--csv-file", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--checkpoint", required=True, help="Path to shared_vision_backbone_best.pt")
    parser.add_argument("--output-dir", default=None, help="Defaults to the checkpoint's parent directory")
    parser.add_argument("--input-size", type=int, nargs=2, default=[128, 128])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Must match the --val-fraction used at training time, so the held-out split lines up",
    )
    parser.add_argument("--num-vis-samples", type=int, default=8, help="Frames to show in the qualitative grid")
    args = parser.parse_args()

    evaluate(args)


if __name__ == "__main__":
    main()
