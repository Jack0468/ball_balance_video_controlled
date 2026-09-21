"""Builds the small Google Colab bundle for the Arm 2 minimal-baseline sweep
(`deployment/arm2_colab_sweep.ipynb`). Run on the DEV MACHINE, not in Colab. Sibling of
`stage_finetune_data_for_colab.ps1`, deliberately much smaller: only the 10 Track 4 sessions
(`telemetry.csv` + `rgb_video.mp4`, ~650MB) plus the handful of code modules the notebook imports
and the ArUco manifest -- NOT the ~4.9GB fine-tuning archive.

Why Python instead of parameterizing the .ps1: the .ps1 relies on `Compress-Archive`, which in
Windows PowerShell 5.1 has a history of writing backslash path separators into zip entries (they
extract as literally-named files on Linux/Colab), and it cannot write a per-file hash manifest or
verify the archive afterwards. `zipfile` writes forward-slash entries, stores the already-compressed
mp4 files uncompressed (deflating video wastes minutes for ~0%), and lets the script self-verify.

Archive layout (single top-level folder, the notebook extracts it to /content/arm2_bundle):
    arm2_bundle/hardware/platform_templates/ground_truth_manifest.json
    arm2_bundle/host_software/ml_jetson_vla/core/{policy_interface,minimal_vlm_policy,vlm_backends,mock_vlm_backends}.py
    arm2_bundle/host_software/ml_jetson_vla/deployment/{qwen_vl_smoke_test,score_minimal_baseline_offline,colab_sweep}.py
    arm2_bundle/host_software/ml_vision/data_processing/auto_label_shared_vision.py   (read-only helper)
    arm2_bundle/host_software/data/01_bronze/session_jetson_track4_*/{telemetry.csv,rgb_video.mp4}
    arm2_bundle/reference/*.json      (local Qwen reference run, original + corrected rescoring)
    arm2_bundle/BUNDLE_MANIFEST.txt   (sha256 + size of every file)

Note the 01_bronze DVC scoped exception (`docs/DATA_STORAGE.md`): if a session directory is not
materialized locally, `dvc pull` it first -- this script fails loudly rather than silently
shipping a partial bundle.

Usage (from anywhere):  python host_software/ml_jetson_vla/deployment/stage_arm2_sweep_for_colab.py [--out PATH]
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import os
import sys
import tempfile
import zipfile
from typing import List, Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_HOST_SOFTWARE_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_REPO_ROOT_DIR = os.path.abspath(os.path.join(_HOST_SOFTWARE_DIR, ".."))

TOP = "arm2_bundle"
EXPECTED_SESSIONS = 10

CODE_FILES: List[str] = [
    "hardware/platform_templates/ground_truth_manifest.json",
    "host_software/ml_jetson_vla/core/policy_interface.py",
    "host_software/ml_jetson_vla/core/minimal_vlm_policy.py",
    "host_software/ml_jetson_vla/core/vlm_backends.py",
    "host_software/ml_jetson_vla/core/mock_vlm_backends.py",
    "host_software/ml_jetson_vla/deployment/qwen_vl_smoke_test.py",
    "host_software/ml_jetson_vla/deployment/score_minimal_baseline_offline.py",
    "host_software/ml_jetson_vla/deployment/colab_sweep.py",
    "host_software/ml_vision/data_processing/auto_label_shared_vision.py",
]
REFERENCE_FILES: List[Tuple[str, str]] = [
    ("host_software/ml_jetson_vla/deployment/arm2_minimal_baseline_prompt_ab_scoring_20260918_RESCORED_v2.json",
     "reference/arm2_local_qwen_20260918_RESCORED_v2.json"),
    ("host_software/ml_jetson_vla/deployment/arm2_minimal_baseline_prompt_ab_scoring_20260918.json",
     "reference/arm2_local_qwen_20260918_ORIGINAL_v1_wrong_frame.json"),
]


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(out_zip: str) -> str:
    entries: List[Tuple[str, str]] = []  # (source path, archive name)
    for rel in CODE_FILES:
        src = os.path.join(_REPO_ROOT_DIR, rel)
        if not os.path.exists(src):
            raise FileNotFoundError(f"required file missing: {src}")
        entries.append((src, f"{TOP}/{rel}"))
    for rel, arc in REFERENCE_FILES:
        src = os.path.join(_REPO_ROOT_DIR, rel)
        if not os.path.exists(src):
            raise FileNotFoundError(f"required reference file missing: {src}")
        entries.append((src, f"{TOP}/{arc}"))

    bronze = os.path.join(_HOST_SOFTWARE_DIR, "data", "01_bronze")
    sessions = sorted(d for d in glob.glob(os.path.join(bronze, "session_jetson_track4_*"))
                      if os.path.isdir(d) and not d.endswith(".dvc"))
    if len(sessions) != EXPECTED_SESSIONS:
        raise RuntimeError(f"expected {EXPECTED_SESSIONS} Track 4 sessions under {bronze}, found "
                           f"{len(sessions)} -- `dvc pull` first if they are not materialized")
    for sdir in sessions:
        name = os.path.basename(sdir)
        for fname in ("telemetry.csv", "rgb_video.mp4"):
            src = os.path.join(sdir, fname)
            if not os.path.exists(src):
                raise FileNotFoundError(f"{src} missing (dvc pull?)")
            entries.append((src, f"{TOP}/host_software/data/01_bronze/{name}/{fname}"))

    manifest_lines = ["sha256  size_bytes  path"]
    total = 0
    for src, arc in entries:
        size = os.path.getsize(src)
        total += size
        manifest_lines.append(f"{_sha256(src)}  {size}  {arc[len(TOP) + 1:]}")

    if os.path.exists(out_zip):
        os.remove(out_zip)
    os.makedirs(os.path.dirname(os.path.abspath(out_zip)), exist_ok=True)
    with zipfile.ZipFile(out_zip, "w") as zf:
        for src, arc in entries:
            ctype = zipfile.ZIP_STORED if src.endswith(".mp4") else zipfile.ZIP_DEFLATED
            zf.write(src, arc, compress_type=ctype)
        zf.writestr(f"{TOP}/BUNDLE_MANIFEST.txt", "\n".join(manifest_lines) + "\n", compress_type=zipfile.ZIP_DEFLATED)

    # Self-verification: CRCs, no backslash entry names, expected file counts.
    with zipfile.ZipFile(out_zip) as zf:
        bad = zf.testzip()
        if bad:
            raise RuntimeError(f"zip integrity check failed at {bad}")
        names = zf.namelist()
    if any("\\" in n for n in names):
        raise RuntimeError("backslash in zip entry names")
    n_video = sum(1 for n in names if n.endswith("rgb_video.mp4"))
    n_csv = sum(1 for n in names if n.endswith("telemetry.csv"))
    if n_video != EXPECTED_SESSIONS or n_csv != EXPECTED_SESSIONS:
        raise RuntimeError(f"unexpected session file counts: {n_video} videos, {n_csv} csvs")

    print(f"Bundle: {out_zip}")
    print(f"  {len(names)} entries, {n_video} sessions, source bytes {total / 1e6:.1f} MB, "
          f"zip {os.path.getsize(out_zip) / 1e6:.1f} MB ({os.path.getsize(out_zip) / 1024 ** 2:.1f} MiB)")
    print(f"  sha256 of the zip: {_sha256(out_zip)}")
    return out_zip


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=os.path.join(tempfile.gettempdir(), "arm2_sweep_bundle.zip"),
                    help="output zip path (default: %%TEMP%%/arm2_sweep_bundle.zip, like the .ps1 sibling)")
    args = ap.parse_args()
    build(args.out)
    print("\nNext (manual): upload it to Google Drive at  MyDrive/vri2026_track4_bootstrap/arm2_sweep_bundle.zip")
    return 0


if __name__ == "__main__":
    sys.exit(main())
