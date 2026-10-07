"""Verify a LeRobot dataset built by convert_to_lerobot.py, then zip it for Colab upload.

Checks the real on-disk dataset (not just that convert_to_lerobot.py exited 0) before trusting it:
schema completeness, the session_episodes.json sidecar against meta/info.json's episode count, and
real per-regime touch_stale/touch_glitch rates (printed, not asserted -- the full dataset's rates are
expected to differ somewhat from the 300-frame dry run's, so a fixed bound here would be guessing).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import zipfile
from typing import Any

import pyarrow.parquet as pq

EXPECTED_FEATURES = {
    "observation.image", "observation.state", "action", "target",
    "regime", "has_real_language_label", "touch_stale", "touch_glitch",
}
KNOWN_REGIMES = {"jetson_track4_rl", "pid_webcam"}


def verify_dataset(root: str) -> dict[str, Any]:
    """Raises AssertionError on a real integrity failure. Returns a summary dict on success."""
    info_path = os.path.join(root, "meta", "info.json")
    assert os.path.exists(info_path), f"missing {info_path}"
    with open(info_path) as f:
        info = json.load(f)
    assert info["codebase_version"] == "v3.0", info["codebase_version"]

    sidecar_path = os.path.join(root, "meta", "session_episodes.json")
    assert os.path.exists(sidecar_path), f"missing {sidecar_path} -- was this built by the current converter?"
    with open(sidecar_path) as f:
        sidecar = json.load(f)["episodes"]
    assert len(sidecar) == info["total_episodes"], (
        f"sidecar has {len(sidecar)} episodes, info.json says {info['total_episodes']}"
    )
    bad_regimes = {e["regime"] for e in sidecar} - KNOWN_REGIMES
    assert not bad_regimes, f"sidecar has unexpected regime values: {bad_regimes}"

    data_dir = os.path.join(root, "data")
    parquet_files = [
        os.path.join(dp, f) for dp, _, files in os.walk(data_dir) for f in files if f.endswith(".parquet")
    ]
    assert parquet_files, f"no parquet files found under {data_dir}"

    # Check schema on the first file before computing any rate -- a missing column must abort
    # immediately, not surface as a confusing KeyError mid-aggregation on some later file.
    # Schema-only read (no row data) -- the first file is still read in full inside the
    # aggregation loop below, so this must not materialize its rows just to look at `.columns`.
    first_cols = pq.read_schema(parquet_files[0]).names
    missing_features = EXPECTED_FEATURES - set(first_cols) - {"observation.image"}  # video, not a parquet column
    assert not missing_features, (
        f"missing expected features in parquet data: {missing_features} -- "
        "was this dataset built before touch_glitch/target were added to the converter?"
    )

    total_rows = 0
    stale_by_regime: dict[str, list[int]] = {r: [0, 0] for r in KNOWN_REGIMES}  # [stale_count, total]
    glitch_by_regime: dict[str, list[int]] = {r: [0, 0] for r in KNOWN_REGIMES}
    for pf in parquet_files:
        cols = pq.read_table(pf).to_pandas()
        total_rows += len(cols)
        for regime, group in cols.groupby("regime"):
            stale_by_regime.setdefault(regime, [0, 0])
            glitch_by_regime.setdefault(regime, [0, 0])
            stale_by_regime[regime][0] += int(group["touch_stale"].sum())
            stale_by_regime[regime][1] += len(group)
            glitch_by_regime[regime][0] += int(group["touch_glitch"].sum())
            glitch_by_regime[regime][1] += len(group)

    assert total_rows == info["total_frames"], (
        f"parquet row count {total_rows} != info.json total_frames {info['total_frames']}"
    )

    summary = {
        "total_episodes": info["total_episodes"],
        "total_frames": info["total_frames"],
        "stale_rate_by_regime": {r: (n / t if t else None) for r, (n, t) in stale_by_regime.items()},
        "glitch_rate_by_regime": {r: (n / t if t else None) for r, (n, t) in glitch_by_regime.items()},
    }
    return summary


def package(root: str, out_zip: str, force: bool) -> str:
    if os.path.exists(out_zip) and not force:
        raise FileExistsError(f"{out_zip} already exists -- pass --force to overwrite")
    root_name = os.path.basename(os.path.normpath(root))
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_STORED) as zf:
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                full = os.path.join(dirpath, fn)
                arcname = os.path.join(root_name, os.path.relpath(full, root))
                zf.write(full, arcname)
    h = hashlib.sha256()
    with open(out_zip, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--out-zip", required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-package", action="store_true", help="verify only, do not zip")
    args = ap.parse_args()

    print(f"verifying {args.dataset_root} ...")
    summary = verify_dataset(args.dataset_root)
    print(json.dumps(summary, indent=2))
    print("VERIFY: OK")

    if args.skip_package:
        return
    print(f"zipping to {args.out_zip} ...")
    sha = package(args.dataset_root, args.out_zip, args.force)
    size = os.path.getsize(args.out_zip)
    print(f"PACKAGE: OK  bytes={size}  sha256={sha}")


if __name__ == "__main__":
    main()
