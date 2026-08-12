"""Combine Dataset 9 (synthetic, generate_synthetic_marker_composites.py) with the
3 real marker sheets (Datasets 5/6/7, sessions 01/02/03 -- NOT Dataset 4/blank,
which was already consumed as the synthetic canvas and would otherwise just dilute
the intentional ratio) into one physically-merged, training-ready 60/40 mix.

Output is a SIBLING of, not nested inside, host_software/data/03_gold/shared_vision/
(Dataset 8, all-real) -- that folder is never touched by this script, and stays the
one evaluate_shared_vision_backbone.py points at, so training (this mix) and
evaluation (Dataset 8) are always physically disjoint files. See
docs/plans -- this task's plan -- Step 5 ("Eval isolation") for the full reasoning.

Physically merges (copies files, concatenates CSVs) rather than teaching the
training script to accept multiple sources, mirroring merge_shared_vision_sessions.py's
existing pattern exactly.

Run as a module from the repo root:

    python -m host_software.ml_vision.data_processing.combine_shared_vision_training_mix \
        --synthetic-dir host_software/data/03_gold/shared_vision_synthetic \
        --output-dir host_software/data/03_gold/shared_vision_synthetic_mix
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from host_software.ml_vision.data_processing.generate_synthetic_marker_composites import MISSING_COMBOS
from host_software.ml_vision.data_processing.merge_shared_vision_sessions import process_session

DEFAULT_REAL_SESSION_DIRS = [
    Path("host_software/data/01_bronze/session_20260810_110239"),  # Dataset 5
    Path("host_software/data/01_bronze/session_20260810_112047"),  # Dataset 6
    Path("host_software/data/01_bronze/session_20260810_114330"),  # Dataset 7
]

# session -> the manifest that actually generated its features (NOT
# ground_truth_manifest.json/aruco_markers_00, which has none) -- needed for the
# combo-balance chart since real sessions don't carry a per-row marker-list column
# the way synthetic rows do via synthetic_markers_json; every frame in a real
# session shows all of that sheet's declared features, so counting is just
# (feature count) x (session row count).
REAL_SESSION_MANIFESTS = {
    "session_20260810_110239": Path("hardware/platform_templates/aruco_markers_01_manifest.json"),
    "session_20260810_112047": Path("hardware/platform_templates/aruco_markers_02_manifest.json"),
    "session_20260810_114330": Path("hardware/platform_templates/aruco_markers_03_manifest.json"),
}


def copy_synthetic(synthetic_dir: Path, output_images_dir: Path, output_masks_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(synthetic_dir / "labels.csv")
    src_images = synthetic_dir / "images"
    src_masks = synthetic_dir / "masks"
    for name in df["image_file"]:
        shutil.copy2(src_images / name, output_images_dir / name)
        shutil.copy2(src_masks / name, output_masks_dir / name)
    return df


def has_missing_combo(markers_json: str) -> bool:
    markers = json.loads(markers_json)
    missing = {f"{s}-{c}" for s, c in MISSING_COMBOS}
    return any(f"{m['shape']}-{m['color']}" in missing for m in markers)


def stratified_sample(df: pd.DataFrame, key_col: str, n: int, seed: int) -> pd.DataFrame:
    """Sample down to n rows while preserving each key_col group's original share --
    so the missing-combo rows (over-generated on purpose) aren't disproportionately
    dropped by a plain random subsample."""
    if n >= len(df):
        return df.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    keys = list(df.groupby(key_col).groups.keys())
    parts: List[pd.DataFrame] = []
    remaining = n
    for i, key in enumerate(keys):
        group = df[df[key_col] == key]
        if i < len(keys) - 1:
            take = min(int(round(len(group) / len(df) * n)), len(group), remaining)
        else:
            take = min(remaining, len(group))
        if take > 0:
            parts.append(group.sample(take, random_state=int(rng.integers(0, 2**31 - 1))))
        remaining -= take
    return pd.concat(parts, ignore_index=True)


def combine(
    synthetic_dir: Path,
    real_session_dirs: List[Path],
    output_dir: Path,
    target_synthetic_fraction: float,
    output_size: Tuple[int, int],
    seed: int,
) -> pd.DataFrame:
    output_images_dir = output_dir / "images"
    output_masks_dir = output_dir / "masks"
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_masks_dir.mkdir(parents=True, exist_ok=True)

    real_frames = [process_session(d, output_images_dir, output_masks_dir, output_size) for d in real_session_dirs]
    real_df = pd.concat(real_frames, ignore_index=True)

    synth_df = copy_synthetic(synthetic_dir, output_images_dir, output_masks_dir)
    synth_df["has_missing_combo"] = synth_df["synthetic_markers_json"].apply(has_missing_combo)

    n_real = len(real_df)
    n_synth_target = round(n_real * target_synthetic_fraction / (1 - target_synthetic_fraction))
    synth_sampled = stratified_sample(synth_df, "has_missing_combo", n_synth_target, seed)

    combined = pd.concat([real_df, synth_sampled], ignore_index=True)
    combined.to_csv(output_dir / "labels.csv", index=False)

    achieved_fraction = len(synth_sampled) / len(combined)
    print(f"Real rows: {n_real}")
    print(f"Synthetic rows available: {len(synth_df)}, target: {n_synth_target}, sampled: {len(synth_sampled)}")
    print(f"Combined total: {len(combined)} (synthetic fraction achieved: {achieved_fraction:.3f}, target: {target_synthetic_fraction})")
    return combined


def plot_combo_balance(combined_df: pd.DataFrame, output_dir: Path) -> None:
    """Bar chart of marker instances per (shape, color) combo across the WHOLE
    combined mix -- the coverage axis plot_coverage.py doesn't measure (that's
    spatial ball position, not marker appearance balance), and the actually
    relevant one for a dataset built specifically to fix appearance imbalance."""
    combo_counts: Dict[str, int] = {}

    for session, manifest_path in REAL_SESSION_MANIFESTS.items():
        n_rows = int((combined_df["session"] == session).sum())
        if n_rows == 0 or not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        for feature in manifest["features"]:
            key = f"{feature['shape']}-{feature['color']}"
            combo_counts[key] = combo_counts.get(key, 0) + n_rows

    synth_rows = combined_df[combined_df["session"] == "synthetic_composite_00"]
    for markers_json in synth_rows.get("synthetic_markers_json", pd.Series(dtype=str)).dropna():
        for m in json.loads(markers_json):
            key = f"{m['shape']}-{m['color']}"
            combo_counts[key] = combo_counts.get(key, 0) + 1

    if not combo_counts:
        return

    missing = {f"{s}-{c}" for s, c in MISSING_COMBOS}
    combos = sorted(combo_counts.keys())
    counts = [combo_counts[c] for c in combos]
    colors = ["darkorange" if c in missing else "steelblue" for c in combos]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(combos, counts, color=colors)
    ax.set_xticks(range(len(combos)))
    ax.set_xticklabels(combos, rotation=45, ha="right")
    ax.set_ylabel("Marker instances in combined training mix")
    ax.set_title("Dataset 9 combo balance (orange = absent from real sheets, blue = present in real sheets)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    out_path = output_dir / "combo_balance_plot.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved combo-balance chart to {out_path}")


def verify(output_dir: Path, expected_total: int) -> None:
    merged_df = pd.read_csv(output_dir / "labels.csv")
    assert len(merged_df) == expected_total, f"labels.csv has {len(merged_df)} rows, expected {expected_total}"

    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    missing_images = [f for f in merged_df["image_file"] if not (images_dir / f).exists()]
    missing_masks = [f for f in merged_df["image_file"] if not (masks_dir / f).exists()]
    assert not missing_images, f"{len(missing_images)} rows missing images, e.g. {missing_images[:5]}"
    assert not missing_masks, f"{len(missing_masks)} rows missing masks, e.g. {missing_masks[:5]}"

    duplicates = merged_df["image_file"][merged_df["image_file"].duplicated()]
    assert duplicates.empty, f"Duplicate image_file after combine: {duplicates.unique().tolist()[:5]}"

    print(f"OK: {len(merged_df)} rows; all images/masks present; no filename collisions.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine synthetic + real sessions into a 60/40 training-ready mix")
    parser.add_argument("--synthetic-dir", type=Path, default=Path("host_software/data/03_gold/shared_vision_synthetic"))
    parser.add_argument("--real-session-dirs", nargs="+", type=Path, default=DEFAULT_REAL_SESSION_DIRS)
    parser.add_argument("--output-dir", type=Path, default=Path("host_software/data/03_gold/shared_vision_synthetic_mix"))
    parser.add_argument("--target-synthetic-fraction", type=float, default=0.6)
    parser.add_argument("--output-size", default="128x128")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} is non-empty; pass --overwrite to replace its contents")

    w_str, h_str = args.output_size.lower().split("x")
    output_size = (int(w_str), int(h_str))

    combined = combine(
        args.synthetic_dir, args.real_session_dirs, args.output_dir,
        args.target_synthetic_fraction, output_size, args.seed,
    )
    print(f"Combined training mix written to {args.output_dir}")

    if not args.skip_verify:
        verify(args.output_dir, len(combined))

    plot_combo_balance(combined, args.output_dir)


if __name__ == "__main__":
    main()
