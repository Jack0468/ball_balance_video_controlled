"""Visual spot-check of shared_vision ball_x_px/ball_y_px labels against the actual
images -- no trained model needed, unlike evaluate_shared_vision_backbone.py's
qualitative grid (which shows predictions, not raw label correctness). Samples
frames spread across every session, overlays the label position and the merged
marker mask directly onto each image, and saves a grid PNG for eyeballing.

Written after the 2026-08-12 ball-label point-reflection bug (auto_label_shared_vision.py
had touch_x/touch_y's sign backwards on both axes) -- re-run this any time the labeling
formula changes, to confirm the fix before spending training compute on it.

Run as a module from the repo root:

    python -m host_software.ml_vision.data_processing.verify_shared_vision_labels \
        --csv-file host_software/data/03_gold/shared_vision/labels.csv \
        --images-dir host_software/data/03_gold/shared_vision/images \
        --masks-dir host_software/data/03_gold/shared_vision/masks
"""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build_grid(
    df: pd.DataFrame,
    images_dir: Path,
    masks_dir: Path,
    samples_per_session: int,
    seed: int,
    annotate_combo_labels: bool = False,
) -> plt.Figure:
    rng = np.random.default_rng(seed)
    rows = []
    for session, group in df.groupby("session"):
        n = min(samples_per_session, len(group))
        rows.append(group.sample(n, random_state=rng.integers(0, 2**31 - 1)))
    sample_df = pd.concat(rows, ignore_index=True).sort_values(["session", "frame_index"]).reset_index(drop=True)

    cols = samples_per_session
    n_sessions = sample_df["session"].nunique()
    fig, axes = plt.subplots(n_sessions, cols, figsize=(3.2 * cols, 3.2 * n_sessions))
    axes = np.atleast_2d(axes)

    for row_idx, (session, group) in enumerate(sample_df.groupby("session")):
        for col_idx, (_, row) in enumerate(group.iterrows()):
            ax = axes[row_idx, col_idx]
            img_path = images_dir / row["image_file"]
            mask_path = masks_dir / row["image_file"]

            img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
            ax.imshow(img)

            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None and mask.max() > 0:
                ax.contour(mask, levels=[127], colors="yellow", linewidths=1.2)

            ax.scatter([row["ball_x_px"]], [row["ball_y_px"]], c="lime", marker="x", s=90, linewidths=2)

            # Opt-in: overlay each synthetic marker's recorded shape/color next to its
            # mask blob, sourced from generate_synthetic_marker_composites.py's
            # synthetic_markers_json column. Catches rendering bugs (wrong hexagon
            # vertices, BGR/RGB swap) before spending training compute. No-ops on
            # real-session rows, which don't have this column.
            markers_json = row.get("synthetic_markers_json")
            if annotate_combo_labels and isinstance(markers_json, str) and markers_json:
                for marker in json.loads(markers_json):
                    cx, cy = marker["center_px"]
                    ax.text(
                        cx, cy - marker.get("size_px", 10) - 3, f"{marker['color']}/{marker['shape']}",
                        color="white", fontsize=5, ha="center",
                        bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.6, linewidth=0),
                    )

            ax.set_title(f"{session}\nframe {row['frame_index']}", fontsize=7)
            ax.axis("off")

        for col_idx in range(len(group), cols):
            axes[row_idx, col_idx].axis("off")

    fig.suptitle("Label Verification: green X = ball_x_px/ball_y_px, yellow = marker mask")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Spot-check shared_vision ball labels against the source images")
    parser.add_argument("--csv-file", type=Path, default=Path("host_software/data/03_gold/shared_vision/labels.csv"))
    parser.add_argument("--images-dir", type=Path, default=Path("host_software/data/03_gold/shared_vision/images"))
    parser.add_argument("--masks-dir", type=Path, default=Path("host_software/data/03_gold/shared_vision/masks"))
    parser.add_argument("--output", type=Path, default=Path("host_software/data/03_gold/shared_vision/label_verification_grid.png"))
    parser.add_argument("--samples-per-session", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--annotate-combo-labels", action="store_true",
        help="Overlay shape/color text labels from the synthetic_markers_json column, "
        "if present (no-op on real-session rows)",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv_file)
    fig = build_grid(df, args.images_dir, args.masks_dir, args.samples_per_session, args.seed, args.annotate_combo_labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    print(f"Saved label verification grid to {args.output}")


if __name__ == "__main__":
    main()
