import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from host_software.ml_vision.training.train_cnn_2d_tracker_marker import SharedVisionBackbone
from host_software.ml_vision.training.augmentations import (
    blur_ops,
    geometric_jitter_ops,
    occlusion_ops,
    photometric_ops,
    shadow_ops,
)


DEFAULT_CSV = Path("host_software/data/03_gold/shared_vision/labels.csv")
DEFAULT_IMAGES_DIR = Path("host_software/data/03_gold/shared_vision/images")
DEFAULT_MASKS_DIR = Path("host_software/data/03_gold/shared_vision/masks")
DEFAULT_OUTPUT_DIR = Path("host_software/ml_vision/experiments/results")

INPUT_SIZE = (128, 128)  # (H, W)

# Same coverage grid as host_software/ml_vision/evaluations/plot_coverage.py, reused
# here so subset sampling and coverage reporting agree on what a "bin" is.
SAFE_X_RANGE = (-80.0, 80.0)
SAFE_Y_RANGE = (-60.0, 60.0)
GRID_SIZE_MM = 10.0


def make_gaussian_heatmap(cx: float, cy: float, size: Tuple[int, int], sigma: float = 5.0) -> torch.Tensor:
    """Mirrors SharedVisionDataset._make_gaussian_heatmap (host_software/ml_vision/training/shared_vision_dataset.py)."""
    h, w = size
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    heatmap = np.exp(-((grid_x - cx) ** 2 + (grid_y - cy) ** 2) / (2 * sigma ** 2))
    return torch.from_numpy(heatmap).unsqueeze(0)


class TrialAugmentedDataset(Dataset):
    """Like SharedVisionDataset, but applies one albumentations transform jointly to
    (image, mask, ball keypoint) so geometric augmentations keep all three in sync.
    Local to this experiment -- production SharedVisionDataset is untouched."""

    def __init__(
        self,
        df: pd.DataFrame,
        images_dir: Path,
        masks_dir: Path,
        transform: A.Compose,
        input_size: Tuple[int, int] = INPUT_SIZE,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.input_size = input_size

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        image_path = self.images_dir / str(row["image_file"])
        mask_path = self.masks_dir / str(row["image_file"])

        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        ball_x = float(row["ball_x_px"])
        ball_y = float(row["ball_y_px"])

        transformed = self.transform(image=image, mask=mask, keypoints=[(ball_x, ball_y)])
        t_image = transformed["image"]
        t_mask = transformed["mask"]
        kp_x, kp_y = transformed["keypoints"][0]

        h, w = self.input_size
        image_tensor = torch.from_numpy(t_image.transpose(2, 0, 1)).float() / 255.0
        mask_tensor = torch.from_numpy(t_mask).float().unsqueeze(0) / 255.0

        ball_xy = torch.tensor([kp_x / w, kp_y / h], dtype=torch.float32)

        # Heatmap target follows production's convention: centred on the (post-augmentation)
        # mask centroid, not the ball -- it's the auxiliary "marker-ness" target for Head 2.
        mask_np = mask_tensor[0].numpy()
        ys_nz, xs_nz = np.where(mask_np > 0.5)
        if len(xs_nz) > 0:
            heatmap_cx, heatmap_cy = float(xs_nz.mean()), float(ys_nz.mean())
        else:
            heatmap_cx, heatmap_cy = w / 2.0, h / 2.0
        heatmap_target = make_gaussian_heatmap(heatmap_cx, heatmap_cy, self.input_size)

        return image_tensor, ball_xy, mask_tensor, heatmap_target


def resize_only() -> A.Compose:
    return A.Compose(
        [A.Resize(*INPUT_SIZE)],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


def build_variants() -> Dict[str, A.Compose]:
    def compose(ops: List[A.BasicTransform]) -> A.Compose:
        return A.Compose(
            [A.Resize(*INPUT_SIZE), *ops],
            keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
        )

    photometric = photometric_ops()
    shadow = shadow_ops()
    geometric = geometric_jitter_ops()
    blur = blur_ops()
    occlusion = occlusion_ops()

    return {
        "baseline": resize_only(),
        "photometric": compose(photometric),
        "shadow": compose(shadow),
        "geometric_jitter": compose(geometric),
        "blur": compose(blur),
        "occlusion": compose(occlusion),
        # Lightweight stand-in for a learned/AutoAugment-style policy -- NOT the
        # RL-searched policy from paper [89]; real policy search is a separate,
        # larger follow-up if this proxy shows the idea has legs.
        "autoaugment_proxy": compose([A.OneOf(photometric + shadow + geometric + blur + occlusion, p=0.9)]),
        "combined": compose(shadow + geometric + blur + occlusion),
    }


def assign_grid_cell(df: pd.DataFrame) -> pd.Series:
    """Bucket each row into the same 10x10mm grid cell plot_coverage.py uses.
    Out-of-safe-zone rows clip to the nearest edge cell rather than being dropped."""
    x_bins = int((SAFE_X_RANGE[1] - SAFE_X_RANGE[0]) / GRID_SIZE_MM)
    y_bins = int((SAFE_Y_RANGE[1] - SAFE_Y_RANGE[0]) / GRID_SIZE_MM)
    x_idx = np.clip(((df["touch_x"] - SAFE_X_RANGE[0]) / GRID_SIZE_MM).astype(int), 0, x_bins - 1)
    y_idx = np.clip(((df["touch_y"] - SAFE_Y_RANGE[0]) / GRID_SIZE_MM).astype(int), 0, y_bins - 1)
    return x_idx * y_bins + y_idx


def load_subset(csv_path: Path, subset_size: int, seed: int) -> pd.DataFrame:
    """Stratified sample across grid cells rather than i.i.d. random sampling.

    Dataset 8 is still centre-heavy even after normalize_spatial_density.py's
    outlier capping (min=50 vs max=284 samples/cell, per dataset_info.md's
    Dataset 8 coverage diagnostics) -- a plain df.sample() would just re-inherit
    that skew into the trial subset, or at small subset sizes could miss sparse
    cells by chance. Round-robin across cells instead: pop one row per occupied
    cell per pass until subset_size is reached, so no cell can dominate.
    """
    df = pd.read_csv(csv_path)

    # A small number of rows have ball_x_px/ball_y_px outside the actual image frame
    # (likely a bad ArUco homography on a handful of source frames -- e.g. one row in
    # Dataset 8 has ball_y_px=284.9 against a 128px-tall image). Harmless in the full
    # 52k-row dataset, but a single one landing in a tiny trial subset/val split can
    # visibly skew the reported px error, so exclude them before sampling.
    in_bounds = (
        (df["ball_x_px"] >= 0) & (df["ball_x_px"] <= INPUT_SIZE[1])
        & (df["ball_y_px"] >= 0) & (df["ball_y_px"] <= INPUT_SIZE[0])
    )
    n_dropped = int((~in_bounds).sum())
    if n_dropped:
        print(f"Dropping {n_dropped} row(s) with ball position outside the {INPUT_SIZE[1]}x{INPUT_SIZE[0]}px frame")
    df = df[in_bounds].reset_index(drop=True)

    if subset_size >= len(df):
        return df.reset_index(drop=True)

    df = df.copy()
    df["_grid_cell"] = assign_grid_cell(df)
    rng = np.random.default_rng(seed)

    cell_pools = {cell: rng.permutation(idxs).tolist() for cell, idxs in df.groupby("_grid_cell").groups.items()}
    cells = list(cell_pools.keys())
    rng.shuffle(cells)

    selected: List[int] = []
    while len(selected) < subset_size and any(cell_pools.values()):
        for cell in cells:
            if len(selected) >= subset_size:
                break
            pool = cell_pools[cell]
            if pool:
                selected.append(pool.pop())

    return df.loc[selected].drop(columns="_grid_cell").reset_index(drop=True)


def split_train_val(df: pd.DataFrame, seed: int, val_fraction: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    indices = torch.randperm(len(df), generator=torch.Generator().manual_seed(seed)).tolist()
    val_size = max(1, int(len(df) * val_fraction))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    return df.iloc[train_indices].reset_index(drop=True), df.iloc[val_indices].reset_index(drop=True)


def train_variant(
    name: str,
    transform: A.Compose,
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

    # Validation always uses resize-only -- we're measuring how well training under
    # this augmentation generalises to clean data, not re-augmented eval noise.
    val_transform = resize_only()

    train_dataset = TrialAugmentedDataset(train_df, images_dir, masks_dir, transform)
    val_dataset = TrialAugmentedDataset(val_df, images_dir, masks_dir, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = SharedVisionBackbone(input_size=INPUT_SIZE).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    criterion_ball = nn.HuberLoss(delta=2.0)
    criterion_mask = nn.BCEWithLogitsLoss()
    criterion_heatmap = nn.MSELoss()

    px_scale = torch.tensor([INPUT_SIZE[1], INPUT_SIZE[0]], device=device, dtype=torch.float32)
    best = {"ball": float("inf"), "mask": float("inf"), "heatmap": float("inf"), "total": float("inf"), "ball_px_error": float("inf")}

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
        ball_px_error_sum, n_samples = 0.0, 0
        with torch.no_grad():
            for images, ball_xy, masks, heatmaps in val_loader:
                images, ball_xy, masks, heatmaps = images.to(device), ball_xy.to(device), masks.to(device), heatmaps.to(device)
                pred_ball_xy, pred_mask_logits, pred_heatmap_logits = model(images)
                val_ball += criterion_ball(pred_ball_xy, ball_xy).item()
                val_mask += criterion_mask(pred_mask_logits, masks).item()
                val_heatmap += criterion_heatmap(torch.sigmoid(pred_heatmap_logits), heatmaps).item()
                n_batches += 1

                px_error = (pred_ball_xy - ball_xy) * px_scale
                ball_px_error_sum += px_error.norm(dim=1).sum().item()
                n_samples += images.size(0)

        val_ball /= max(n_batches, 1)
        val_mask /= max(n_batches, 1)
        val_heatmap /= max(n_batches, 1)
        val_total = val_ball + 0.5 * val_mask + 0.1 * val_heatmap
        ball_px_error = ball_px_error_sum / max(n_samples, 1)

        print(f"  [{name}] epoch {epoch + 1}/{epochs} | val_total={val_total:.4f} ball_px_err={ball_px_error:.2f}px")

        if val_total < best["total"]:
            best = {
                "ball": val_ball,
                "mask": val_mask,
                "heatmap": val_heatmap,
                "total": val_total,
                "ball_px_error": ball_px_error,
            }

    return best


def run_sweep(args: argparse.Namespace) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = load_subset(args.csv_file, args.subset_size, args.seed)
    train_df, val_df = split_train_val(df, args.seed)
    print(f"Subset: {len(df)} rows ({len(train_df)} train / {len(val_df)} val)")

    all_variants = build_variants()
    variant_names = args.variants if args.variants else list(all_variants.keys())
    unknown = set(variant_names) - set(all_variants.keys())
    if unknown:
        raise ValueError(f"Unknown variant(s): {sorted(unknown)}. Available: {list(all_variants.keys())}")

    results = []
    for name in variant_names:
        print(f"\nTraining variant: {name}")
        metrics = train_variant(
            name,
            all_variants[name],
            train_df,
            val_df,
            args.images_dir,
            args.masks_dir,
            args.epochs,
            args.batch_size,
            args.lr,
            device,
            args.seed,
        )
        results.append({"variant": name, **metrics})

    return pd.DataFrame(results).sort_values("total").reset_index(drop=True)


def save_report(results_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = output_dir / f"augmentation_trial_{timestamp}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved comparison table to {csv_path}")
    print(results_df.to_string(index=False))

    plt.figure(figsize=(8, 5))
    plt.bar(results_df["variant"], results_df["ball_px_error"], color="steelblue")
    plt.ylabel("Best val ball position error (px)")
    plt.title("Augmentation Strategy Comparison")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plot_path = output_dir / f"augmentation_trial_{timestamp}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved comparison chart to {plot_path}")


def aggregate_seed_runs(runs: List[pd.DataFrame]) -> pd.DataFrame:
    """Average each variant's metrics across seeds -- a single seed's ranking can flip
    on noise alone (see e.g. baseline's 28px-vs-52px swing across epochs in one run)."""
    combined = pd.concat(runs, ignore_index=True)
    agg = combined.groupby("variant").agg(
        ball_px_error_mean=("ball_px_error", "mean"),
        ball_px_error_std=("ball_px_error", "std"),
        total_mean=("total", "mean"),
        total_std=("total", "std"),
        ball_mean=("ball", "mean"),
        mask_mean=("mask", "mean"),
        heatmap_mean=("heatmap", "mean"),
        n_seeds=("total", "count"),
    )
    return agg.sort_values("total_mean").reset_index()


def save_multi_seed_report(agg_df: pd.DataFrame, raw_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary_path = output_dir / f"augmentation_trial_multiseed_{timestamp}_summary.csv"
    raw_path = output_dir / f"augmentation_trial_multiseed_{timestamp}_raw.csv"
    agg_df.to_csv(summary_path, index=False)
    raw_df.to_csv(raw_path, index=False)
    print(f"\nSaved multi-seed summary to {summary_path}")
    print(f"Saved per-seed raw results to {raw_path}")
    print(agg_df.to_string(index=False))

    plt.figure(figsize=(9, 5))
    plt.bar(
        agg_df["variant"],
        agg_df["ball_px_error_mean"],
        yerr=agg_df["ball_px_error_std"].fillna(0.0),
        capsize=4,
        color="steelblue",
    )
    plt.ylabel("Val ball position error (px), mean ± std across seeds")
    plt.title(f"Augmentation Strategy Comparison ({int(agg_df['n_seeds'].iloc[0])} seeds)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plot_path = output_dir / f"augmentation_trial_multiseed_{timestamp}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved comparison chart to {plot_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare augmentation strategies on a small subset of Dataset 8 (merged shared-vision data)"
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
        "--num-seeds",
        type=int,
        default=1,
        help="Run the full sweep this many times (seed, seed+1, ...) and average results, "
        "since a single seed's ranking can flip on noise alone",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="Subset of variant names to run (default: all -- baseline, photometric, shadow, "
        "geometric_jitter, blur, occlusion, autoaugment_proxy, combined)",
    )
    args = parser.parse_args()

    if not args.csv_file.exists():
        raise FileNotFoundError(
            f"CSV not found: {args.csv_file}. Run merge_shared_vision_sessions.py first."
        )

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
