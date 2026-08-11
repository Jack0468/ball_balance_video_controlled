import argparse
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


DEFAULT_SESSION_DIRS = [
    Path("host_software/data/01_bronze/session_20260810_104132"),  # Dataset 4 (blank platform)
    Path("host_software/data/01_bronze/session_20260810_110239"),  # Dataset 5 (5 colour circles)
    Path("host_software/data/01_bronze/session_20260810_112047"),  # Dataset 6 (4 blue shapes)
    Path("host_software/data/01_bronze/session_20260810_114330"),  # Dataset 7 (5 mixed shape/colour)
]
DEFAULT_OUTPUT_DIR = Path("host_software/data/03_gold/shared_vision")

FRAME_MASK_RE = re.compile(r"^(frame_\d+)_.+_mask\.png$")


def prefixed_filename(session_name: str, orig_filename: str) -> str:
    """Session-qualify a filename so identical per-session names (e.g. frame_00000.png,
    which every session restarts numbering from) never collide once flattened together."""
    return f"{session_name}__{orig_filename}"


def index_masks_by_frame(mask_dir: Path) -> Dict[str, List[Path]]:
    """Group this session's per-feature mask files by frame stem, in one pass over mask_dir."""
    index: Dict[str, List[Path]] = defaultdict(list)
    for mask_path in mask_dir.iterdir():
        match = FRAME_MASK_RE.match(mask_path.name)
        if not match:
            continue
        index[match.group(1)].append(mask_path)
    return index


def build_combined_mask(mask_paths: List[Path], size: Tuple[int, int] = (128, 128)) -> np.ndarray:
    """Union all per-feature binary masks for one frame into a single mask.

    Masks are 0/255, so np.maximum-folding is equivalent to a bitwise OR. An
    empty mask_paths list (e.g. every frame in the blank-platform session,
    which has no features to render) correctly yields an all-zero mask -- a
    real 'no markers present' negative example, not an error.
    """
    h, w = size
    combined = np.zeros((h, w), dtype=np.uint8)
    for mask_path in mask_paths:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"  Warning: could not read mask {mask_path}, skipping")
            continue
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        combined = np.maximum(combined, mask)
    return combined


def validate_session_ready(session_dir: Path) -> Tuple[Path, Path, Path]:
    """Confirm a session's auto-labeling run actually finished before merging it.

    Raises rather than silently skipping -- dropping a session quietly would
    shrink the training pool without anyone noticing.
    """
    csv_path = session_dir / "shared_vision_labels.csv"
    images_cropped_dir = session_dir / "images_cropped"
    masks_dir = session_dir / "masks"

    missing = [str(p) for p in (csv_path, images_cropped_dir, masks_dir) if not p.exists()]
    if missing:
        raise RuntimeError(
            f"Session {session_dir} is not ready to merge -- missing: {missing}. "
            "auto_label_shared_vision.py likely hasn't finished for this session "
            "(or hasn't been run at all). Wait for it to complete, or remove this "
            "session from --session-dirs."
        )
    return csv_path, images_cropped_dir, masks_dir


def process_session(
    session_dir: Path,
    output_images_dir: Path,
    output_masks_dir: Path,
    output_size: Tuple[int, int],
) -> pd.DataFrame:
    """Copy one session's images, write its combined masks, and return its labels
    with image_file rewritten to the merged (session-prefixed) naming scheme."""
    csv_path, images_cropped_dir, masks_dir = validate_session_ready(session_dir)
    df = pd.read_csv(csv_path)
    session_name = session_dir.name
    mask_index = index_masks_by_frame(masks_dir)

    missing_images = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=session_name):
        orig_name = str(row["image_file"])
        src_img = images_cropped_dir / orig_name
        if not src_img.exists():
            missing_images.append(orig_name)
            continue

        new_name = prefixed_filename(session_name, orig_name)
        shutil.copy2(src_img, output_images_dir / new_name)

        stem = Path(orig_name).stem
        combined_mask = build_combined_mask(mask_index.get(stem, []), size=output_size)
        cv2.imwrite(str(output_masks_dir / new_name), combined_mask)

    if missing_images:
        raise RuntimeError(
            f"[{session_name}] {len(missing_images)} CSV row(s) reference images_cropped "
            f"files that don't exist on disk (e.g. {missing_images[:5]}). This session's "
            "auto-labeling run may be incomplete -- refusing to merge a partial session."
        )

    df["orig_image_file"] = df["image_file"]
    df["session"] = session_name
    df["image_file"] = df["orig_image_file"].apply(lambda name: prefixed_filename(session_name, str(name)))
    return df


def merge_sessions(
    session_dirs: List[Path],
    output_dir: Path,
    output_size: Tuple[int, int] = (128, 128),
    overwrite: bool = False,
) -> pd.DataFrame:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"{output_dir} is non-empty; pass --overwrite to replace its contents")

    output_images_dir = output_dir / "images"
    output_masks_dir = output_dir / "masks"
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_masks_dir.mkdir(parents=True, exist_ok=True)

    session_frames = [
        process_session(session_dir, output_images_dir, output_masks_dir, output_size)
        for session_dir in session_dirs
    ]
    merged_df = pd.concat(session_frames, ignore_index=True)
    merged_df.to_csv(output_dir / "labels.csv", index=False)
    return merged_df


def verify_merge(output_dir: Path, session_dirs: List[Path]) -> None:
    """Re-reads everything from disk (rather than trusting in-memory state) so this
    also catches bugs where what got written diverged from what merge_sessions() thinks it wrote."""
    merged_df = pd.read_csv(output_dir / "labels.csv")

    expected_total = 0
    print("Per-session row counts:")
    for session_dir in session_dirs:
        source_df = pd.read_csv(session_dir / "shared_vision_labels.csv")
        expected_total += len(source_df)
        session_rows = merged_df[merged_df["session"] == session_dir.name]
        print(f"  {session_dir.name}: {len(session_rows)} merged (source had {len(source_df)})")
        assert len(session_rows) == len(source_df), (
            f"{session_dir.name}: merged {len(session_rows)} rows != source {len(source_df)} rows"
        )

    assert len(merged_df) == expected_total, f"Merged total {len(merged_df)} != expected {expected_total}"

    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    missing_images = [f for f in merged_df["image_file"] if not (images_dir / f).exists()]
    missing_masks = [f for f in merged_df["image_file"] if not (masks_dir / f).exists()]
    assert not missing_images, f"{len(missing_images)} rows missing images, e.g. {missing_images[:5]}"
    assert not missing_masks, f"{len(missing_masks)} rows missing masks, e.g. {missing_masks[:5]}"

    duplicates = merged_df["image_file"][merged_df["image_file"].duplicated()]
    assert duplicates.empty, f"Duplicate image_file after merge: {duplicates.unique().tolist()[:5]}"

    print(
        f"OK: {len(merged_df)} rows across {len(session_dirs)} sessions; "
        "all images/masks present; no filename collisions."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge auto-labeled shared-vision sessions into one 03_gold training-ready dataset"
    )
    parser.add_argument(
        "--session-dirs",
        nargs="+",
        type=Path,
        default=DEFAULT_SESSION_DIRS,
        help="Session dirs to merge (default: the 4 confirmed 2026-08-10 sessions)",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--output-size",
        default="128x128",
        help="Combined-mask resolution WxH (default: 128x128, mirrors auto_label_shared_vision.py)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into a non-empty --output-dir")
    parser.add_argument("--skip-verify", action="store_true", help="Skip the post-merge verification pass")
    args = parser.parse_args()

    for session_dir in args.session_dirs:
        if not session_dir.exists():
            raise FileNotFoundError(f"Session dir not found: {session_dir}")

    w_str, h_str = args.output_size.lower().split("x")
    output_size = (int(w_str), int(h_str))

    merge_sessions(args.session_dirs, args.output_dir, output_size, args.overwrite)
    print(f"Merged dataset written to {args.output_dir}")

    if not args.skip_verify:
        verify_merge(args.output_dir, args.session_dirs)


if __name__ == "__main__":
    main()
