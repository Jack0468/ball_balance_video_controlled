"""Verify a LeRobot dataset built by convert_to_lerobot.py, then zip it for Colab upload.

Checks the real on-disk dataset (not just that convert_to_lerobot.py exited 0) before trusting it:
schema completeness, the session_episodes.json sidecar against meta/info.json's episode count, and
real per-regime touch_stale/touch_glitch rates (printed, not asserted -- the full dataset's rates are
expected to differ somewhat from the 300-frame dry run's, so a fixed bound here would be guessing).

**Does NOT decode video by default.** The schema/row-count checks above never open a single .mp4 --
confirmed as a real gap on 2026-10-13, after a `lerobot_full_2026_10_07` dataset that this script had
already verified+zipped (docs/PROJECT_LOGBOOK.md, 2026-10-07/08 entry) later crashed torchcodec mid-decode
during an eval DataLoader pass on Colab. Pass `--check-video-decode` (`verify_video_decode()` below) to
additionally decode every video file start-to-finish with PyAV and catch a corrupt/truncated file before
it reaches Colab. See the 2026-10-13 logbook entry for why this is opt-in rather than the default (slow:
full-file CPU decode of every video; also a different decoder than torchcodec, so a pass here is not a
torchcodec-safety guarantee -- see that function's docstring).
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


def verify_video_decode(root: str, info: dict[str, Any]) -> dict[str, Any]:
    """Decode every video file this dataset references, start-to-finish, with PyAV (software libdav1d).

    Catches a corrupt/truncated video file that `verify_dataset`'s schema/row-count checks cannot see
    (they never open a .mp4). Raises AssertionError naming the exact file and how many good frames
    decoded before failure.

    Important caveat: this uses PyAV, not torchcodec (torchcodec is not installed in this project's
    pinned Windows env, `ball_balance_env`). PyAV's libdav1d AV1 decoder is software and has historically
    been more fault-tolerant than torchcodec's bundled FFmpeg AV1 path (and than any hardware/NVDEC AV1
    decoder torchcodec may select on a Colab GPU). A clean pass here only rules out gross file-level
    corruption/truncation -- it is not a guarantee the same bytes will decode cleanly under torchcodec.
    """
    import av  # local import: optional dependency, only needed for this check

    video_dir = os.path.join(root, "videos")
    assert os.path.isdir(video_dir), f"missing {video_dir}"
    video_files = sorted(
        os.path.join(dp, f) for dp, _, files in os.walk(video_dir) for f in files if f.endswith(".mp4")
    )
    assert video_files, f"no .mp4 files found under {video_dir}"

    total_decoded_frames = 0
    per_file_frames: dict[str, int] = {}
    for vf in video_files:
        rel = os.path.relpath(vf, root)
        n = 0
        try:
            container = av.open(vf)  # can itself raise (e.g. truncated moov atom) before any decode
            try:
                stream = container.streams.video[0]
                for packet in container.demux(stream):
                    n += len(packet.decode())
            finally:
                container.close()
        except Exception as exc:  # noqa: BLE001 -- surface exactly which file/how far it got
            raise AssertionError(
                f"video decode failed in {rel} after {n} good frame(s): {type(exc).__name__}: {exc}"
            ) from exc
        per_file_frames[rel] = n
        total_decoded_frames += n

    assert total_decoded_frames == info["total_frames"], (
        f"decoded {total_decoded_frames} frames across {len(video_files)} video file(s), "
        f"but info.json total_frames is {info['total_frames']}"
    )
    return {"video_files_checked": len(video_files), "total_decoded_frames": total_decoded_frames,
            "per_file_frames": per_file_frames}


def verify_dataset(root: str, check_video_decode: bool = False) -> dict[str, Any]:
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
    if check_video_decode:
        summary["video_decode"] = verify_video_decode(root, info)
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
    ap.add_argument("--check-video-decode", action="store_true",
                     help="decode every video file start-to-finish with PyAV (slow; catches corrupt/"
                          "truncated video files that schema+row-count checks cannot see -- see "
                          "verify_video_decode() docstring for the torchcodec caveat)")
    args = ap.parse_args()

    print(f"verifying {args.dataset_root} ...")
    summary = verify_dataset(args.dataset_root, check_video_decode=args.check_video_decode)
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
