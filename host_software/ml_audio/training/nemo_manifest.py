"""Build NeMo-format training manifests from training_v2/.

NeMo's EncDecClassificationModel expects one JSON object per line:
    {"audio_filepath": "...", "duration": 1.25, "label": "go_red"}

Kept as a small, standalone, NeMo-free module (only needs soundfile) so it
can be tested locally before ever touching Colab -- manifest-building has
nothing to do with NeMo itself, no reason to require its Linux-only
dependencies just to write out some JSON lines. The Colab fine-tuning
notebook imports this directly, matching the rest of this plan's convention
of reusing the exact same code locally and in Colab rather than
duplicating logic into notebook cells.

Usage:
    python nemo_manifest.py
    python nemo_manifest.py --dataset-root <path> --out-dir <path>
"""

import argparse
import json
import os
import sys

import soundfile as sf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ML_AUDIO_DIR = os.path.dirname(SCRIPT_DIR)
HOST_SOFTWARE_DIR = os.path.dirname(ML_AUDIO_DIR)
if HOST_SOFTWARE_DIR not in sys.path:
    sys.path.insert(0, HOST_SOFTWARE_DIR)

from ml_audio.evaluations.evaluate_audio_classifier import gather_labeled_files  # noqa: E402
from ml_audio.training.train_audio_command_classifier import (  # noqa: E402
    DEFAULT_DATASET_ROOT,
    discover_labels,
)


def build_manifest_entries(dataset_root: str, split: str, labels: list[str]) -> list[dict]:
    split_dir = os.path.join(dataset_root, split)
    labeled_files = gather_labeled_files(split_dir, labels)
    entries = []
    for label, wav_path in labeled_files:
        info = sf.info(wav_path)
        duration = info.frames / info.samplerate
        entries.append({
            "audio_filepath": os.path.abspath(wav_path),
            "duration": round(duration, 4),
            "label": label,
        })
    return entries


def write_manifest(entries: list[dict], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build NeMo-format train/val manifests from training_v2/."
    )
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--out-dir", default=os.path.join(ML_AUDIO_DIR, "training", "nemo_manifests"))
    args = parser.parse_args()

    labels = discover_labels(args.dataset_root)
    print(f"Labels ({len(labels)}, alphabetical -- matches labels.json convention used everywhere else "
          f"in this pipeline): {labels}")

    for split in ("train", "val"):
        entries = build_manifest_entries(args.dataset_root, split, labels)
        out_path = os.path.join(args.out_dir, f"{split}_manifest.json")
        write_manifest(entries, out_path)
        durations = [e["duration"] for e in entries]
        print(f"{split}: {len(entries)} entries -> {out_path} "
              f"(duration range {min(durations):.2f}-{max(durations):.2f}s)")

        counts: dict[str, int] = {}
        for e in entries:
            counts[e["label"]] = counts.get(e["label"], 0) + 1
        for label in labels:
            print(f"    {label:14s} {counts.get(label, 0)}")


if __name__ == "__main__":
    main()
