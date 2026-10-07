"""Builds the Colab grounding bundle for `training/colab/qwen_grounding_colab.ipynb`.

For every Track 4 session (`host_software/data/01_bronze/session_jetson_track4_*`) this keeps:
  - every `--stride`-th frame (by `frame_index`), plus
  - every frame whose `audio_command` is a directional/hold/stop command (`DIRECTIONAL_KEEP`),
so directional commands are covered in full and colour commands are stride-sampled.

Each kept frame is written as a JPEG (quality 92) under `<out>/bundle/images/`, with one row in
`<out>/bundle/manifest.csv`. The per-frame ArUco homography (mm -> raw px, the same
`estimate_homography_from_aruco` the scorer uses) is stored alongside, because the Colab side
has no ArUco-detection step of its own and the scorer cannot convert predicted pixels to mm
without it. Frames with no homography are kept but flagged empty, so the motor test still scores.

Outputs (relative to `--out-dir`, default `host_software/data/colab_bundle`):
  - `grounding_bundle.zip`   zip whose root folder is `bundle/`
  - `BUNDLE_MANIFEST.txt`    OUTSIDE the zip (a zip cannot contain its own sha256): zip sha256,
                             size, stride and per-command frame counts

Telemetry rows with NaN in any required column, and kept frames with no decodable video frame,
are skipped and counted; they are never filled.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

import av
import cv2
import numpy as np
import pandas as pd

_THIS_FILE = Path(__file__).resolve()
_ML_JETSON_VLA_DIR = _THIS_FILE.parents[1]
_HOST_SOFTWARE_DIR = _THIS_FILE.parents[2]
_REPO_ROOT_DIR = _THIS_FILE.parents[3]
for _p in (_HOST_SOFTWARE_DIR, _REPO_ROOT_DIR):
    if str(_p) not in sys.path:
        sys.path.append(str(_p))

from ml_jetson_vla.deployment.score_minimal_baseline_offline import (  # noqa: E402
    GROUND_TRUTH_MANIFEST,
    compute_transition_flags,
)
from host_software.ml_vision.data_processing.auto_label_shared_vision import (  # noqa: E402
    build_aruco_lookup,
    estimate_homography_from_aruco,
    load_manifest_full,
)
from host_software.src.touch_ground_truth_filter import flag_touch_position_outliers  # noqa: E402

DEFAULT_STRIDE = 30
DEFAULT_DATA_ROOT = _HOST_SOFTWARE_DIR / "data" / "01_bronze"
DEFAULT_OUT_DIR = _HOST_SOFTWARE_DIR / "data" / "colab_bundle"
SESSION_GLOB = "session_jetson_track4_*"
ZIP_NAME = "grounding_bundle.zip"
BUNDLE_MANIFEST_NAME = "BUNDLE_MANIFEST.txt"
JPEG_QUALITY = 92
DIRECTIONAL_KEEP = frozenset({"forward", "left", "right", "backward", "hold", "stop"})
REQUIRED_COLUMNS = (
    "frame_index", "host_timestamp_ms", "touch_x", "touch_y", "theta_a", "theta_b", "theta_c",
    "target_x", "target_y", "audio_command",
)
MANIFEST_COLUMNS = (
    "image_file", "session", "frame_index", "audio_command", "touch_x", "touch_y",
    "theta_a", "theta_b", "theta_c", "target_x", "target_y", "host_timestamp_ms",
    "target_transition", "touch_suspect", "homography_mm_to_px",
)


def iter_wanted_frames(video_path: Path, wanted: Set[int]) -> Iterator[Tuple[int, np.ndarray]]:
    """Streaming variant of the scorer's `read_frames_at_indices`: yields (index, bgr) one
    frame at a time instead of holding every requested frame in RAM (directional commands
    make up most of a session, so the scorer's version would not fit)."""
    remaining = set(wanted)
    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        for i, frame in enumerate(container.decode(stream)):
            if not remaining:
                break
            if i in remaining:
                remaining.discard(i)
                yield i, frame.to_ndarray(format="bgr24")
    finally:
        container.close()


def is_row_complete(row: pd.Series) -> bool:
    return not bool(row[list(REQUIRED_COLUMNS)].isna().any())


def select_candidate_frames(df: pd.DataFrame, stride: int) -> List[int]:
    """Row positions (not frame_index values) that match the stride or directional rule."""
    positions: List[int] = []
    for pos, (frame_index, command) in enumerate(zip(df["frame_index"], df["audio_command"])):
        is_stride = int(frame_index) % stride == 0
        is_directional = isinstance(command, str) and command in DIRECTIONAL_KEEP
        if is_stride or is_directional:
            positions.append(pos)
    return positions


def homography_to_json(h: Optional[np.ndarray]) -> str:
    if h is None:
        return ""
    return json.dumps(np.asarray(h, dtype=float).tolist())


def build_session_rows(
    session_dir: Path,
    stride: int,
    aruco_lookup: Dict[int, List[float]],
    images_dir: Path,
    skip_counts: Counter,
) -> List[Dict[str, object]]:
    session = session_dir.name
    df = pd.read_csv(session_dir / "telemetry.csv")
    transition = compute_transition_flags(df)
    suspect = flag_touch_position_outliers(df["touch_x"], df["touch_y"], touch_valid=None)

    positions = select_candidate_frames(df, stride)
    wanted: Set[int] = set()
    valid_positions: List[int] = []
    for pos in positions:
        row = df.iloc[pos]
        if not is_row_complete(row):
            skip_counts["skipped_nan_telemetry"] += 1
            continue
        valid_positions.append(pos)
        wanted.add(int(row["frame_index"]))

    pos_by_frame = {int(df.iloc[p]["frame_index"]): p for p in valid_positions}
    video_path = session_dir / "rgb_video.mp4"
    rows: List[Dict[str, object]] = []
    decoded: Set[int] = set()
    for frame_index, frame_bgr in iter_wanted_frames(video_path, wanted):
        decoded.add(frame_index)
        pos = pos_by_frame[frame_index]
        row = df.iloc[pos]
        h = estimate_homography_from_aruco(frame_bgr, aruco_lookup)
        if h is None:
            skip_counts["kept_without_homography"] += 1
        image_file = f"{session}__f{frame_index:06d}.jpg"
        cv2.imwrite(str(images_dir / image_file), frame_bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        rows.append({
            "image_file": f"images/{image_file}",
            "session": session,
            "frame_index": frame_index,
            "audio_command": row["audio_command"],
            "touch_x": row["touch_x"],
            "touch_y": row["touch_y"],
            "theta_a": row["theta_a"],
            "theta_b": row["theta_b"],
            "theta_c": row["theta_c"],
            "target_x": row["target_x"],
            "target_y": row["target_y"],
            "host_timestamp_ms": int(row["host_timestamp_ms"]),
            "target_transition": bool(transition.iloc[pos]),
            "touch_suspect": bool(suspect.iloc[pos]),
            "homography_mm_to_px": homography_to_json(h),
        })
    skip_counts["skipped_missing_video_frame"] += len(wanted - decoded)
    return rows


def write_manifest(rows: List[Dict[str, object]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(MANIFEST_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in MANIFEST_COLUMNS})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def zip_bundle(bundle_dir: Path, zip_path: Path) -> None:
    """Zips with the `bundle/` folder as the archive root. JPEGs are already compressed,
    so ZIP_STORED avoids wasted CPU."""
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for path in sorted(bundle_dir.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=str(Path("bundle") / path.relative_to(bundle_dir)).replace(os.sep, "/"))


def write_bundle_manifest(
    out_dir: Path, zip_path: Path, stride: int, sessions: List[str],
    rows: List[Dict[str, object]], skip_counts: Counter,
) -> Path:
    per_command = Counter(str(r["audio_command"]) for r in rows)
    lines = [
        f"created_utc: {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}",
        f"zip: {zip_path.name}",
        f"zip_sha256: {sha256_file(zip_path)}",
        f"zip_bytes: {zip_path.stat().st_size}",
        f"stride: {stride}",
        f"jpeg_quality: {JPEG_QUALITY}",
        f"sessions: {len(sessions)}",
        f"total_frames: {len(rows)}",
        "frames_per_command:",
    ]
    lines += [f"  {cmd}: {n}" for cmd, n in sorted(per_command.items())]
    lines.append("skip_and_flag_counts:")
    lines += [f"  {k}: {v}" for k, v in sorted(skip_counts.items())]
    path = out_dir / BUNDLE_MANIFEST_NAME
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run(stride: int, data_root: Path, out_dir: Path, only_session: Optional[str]) -> None:
    if stride < 1:
        raise ValueError("stride must be >= 1")
    sessions = sorted(p for p in data_root.glob(SESSION_GLOB) if p.is_dir() and p.suffix != ".dvc")
    if only_session:
        sessions = [p for p in sessions if p.name == only_session]
        if not sessions:
            raise FileNotFoundError(f"no Track 4 session named {only_session!r} under {data_root}")
    if not sessions:
        raise FileNotFoundError(f"no sessions matching {SESSION_GLOB} under {data_root}")

    aruco_markers, _, _, _ = load_manifest_full(Path(GROUND_TRUTH_MANIFEST))
    aruco_lookup = build_aruco_lookup(aruco_markers)

    bundle_dir = out_dir / "bundle"
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    images_dir = bundle_dir / "images"
    images_dir.mkdir(parents=True)

    skip_counts: Counter = Counter()
    rows: List[Dict[str, object]] = []
    for session_dir in sessions:
        session_rows = build_session_rows(session_dir, stride, aruco_lookup, images_dir, skip_counts)
        print(f"{session_dir.name}: kept {len(session_rows)} frames")
        rows.extend(session_rows)

    write_manifest(rows, bundle_dir / "manifest.csv")
    zip_path = out_dir / ZIP_NAME
    zip_bundle(bundle_dir, zip_path)
    write_bundle_manifest(out_dir, zip_path, stride, [s.name for s in sessions], rows, skip_counts)

    per_command = Counter(str(r["audio_command"]) for r in rows)
    print(f"\nstride={stride} sessions={len(sessions)} total_frames={len(rows)}")
    print("frames per command:")
    for cmd, n in sorted(per_command.items()):
        print(f"  {cmd:10s} {n}")
    print("skip/flag counts:", dict(skip_counts))
    print(f"zip: {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")
    print(f"zip sha256: {sha256_file(zip_path)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE,
                        help=f"keep every N-th frame by frame_index (default {DEFAULT_STRIDE})")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--session", type=str, default=None,
                        help="process only this session folder name (for a quick test run)")
    args = parser.parse_args()
    run(args.stride, args.data_root, args.out_dir, args.session)


if __name__ == "__main__":
    main()
