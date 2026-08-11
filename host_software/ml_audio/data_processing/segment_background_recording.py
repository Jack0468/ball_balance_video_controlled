"""Segment a long, unlabeled background recording into fixed-length
_background_ training/val clips.

Built for the ~23-minute general-lab-sounds recording at
data/01_background_noise/lab_background_sound_01.wav (see
docs/plans/audio_eval_notebook_refactor_plan.md, "Larger Background/
Noise-Profile Source" section) -- a much richer, more diverse background
source than the ~90 real-recording clips (from just 2 source files,
robot_background_sound[_01].wav) that make up the vast majority-synthetic
existing _background_ class. Cross-checked against the dataset on disk: of
the 1290 train + 240 val _background_ clips, only 90 (all in train, none in
val) come from real recordings at all -- the rest are near-silent synthetic
TTS renders. So today the eval set for the _background_ class contains zero
real ambient noise, and the live-stream bottleneck this whole effort is
chasing (background dominating almost every window of a noisy continuous
stream) is invisible to that eval set by construction. This script's train/
val split deliberately includes real-noise segments in val as well, unlike
the existing bgreal_* clips.

Output windows match the model's fixed input exactly (16 kHz mono, 1.25s /
20000 samples -- same as OUTPUT_SEQUENCE_LENGTH in audio_receiver_pytorch.py)
and follow the existing bgreal_<source>__<split>__NNNNNN.wav naming
convention already used for robot_background_sound[_01].wav, so they drop
into training_v2/{train,val}/_background_/ as more of the same thing, not a
new format.

Segments that fail the same empty-clip energy gate used everywhere else in
this pipeline (align_speech_to_fixed_length in audio_receiver_pytorch.py:
peak < 0.03 or rms < 0.003) are dropped -- keeping them would just add more
near-silent filler on top of the 1200 synthetic clips already dominated by
exactly that problem, defeating the point of this source.

Usage:
    python segment_background_recording.py
    python segment_background_recording.py --source <wav> --val-fraction 0.15
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone

import numpy as np
import soundfile as sf
from scipy.signal import resample

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ML_AUDIO_DIR = os.path.dirname(SCRIPT_DIR)
HOST_SOFTWARE_DIR = os.path.dirname(ML_AUDIO_DIR)
if HOST_SOFTWARE_DIR not in sys.path:
    sys.path.insert(0, HOST_SOFTWARE_DIR)

from ml_audio.audio_receiver_pytorch import OUTPUT_SEQUENCE_LENGTH, SAMPLE_RATE  # noqa: E402

DEFAULT_SOURCE = os.path.join(
    ML_AUDIO_DIR, "data", "01_background_noise", "lab_background_sound_01.wav"
)
DEFAULT_DATASET_ROOT = os.path.join(
    ML_AUDIO_DIR, "data", "synthetic+real_dataset_large", "training_v2"
)
REPORT_DIR = os.path.join(SCRIPT_DIR, "reports")

# Same gate as align_speech_to_fixed_length() in audio_receiver_pytorch.py.
EMPTY_PEAK_THRESHOLD = 0.03
EMPTY_RMS_THRESHOLD = 0.003


def load_mono_16k(source_path: str) -> np.ndarray:
    audio, sr = sf.read(source_path, dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != SAMPLE_RATE:
        print(f"Resampling from {sr} Hz to {SAMPLE_RATE} Hz...")
        num_samples = int(len(audio) * SAMPLE_RATE / sr)
        audio = resample(audio, num_samples).astype(np.float32)
    return audio


def segment(audio: np.ndarray, window_samples: int) -> list[np.ndarray]:
    n_windows = len(audio) // window_samples
    return [
        audio[i * window_samples:(i + 1) * window_samples]
        for i in range(n_windows)
    ]


def is_empty(clip: np.ndarray) -> bool:
    peak = np.max(np.abs(clip))
    rms = np.sqrt(np.mean(clip**2))
    return peak < EMPTY_PEAK_THRESHOLD or rms < EMPTY_RMS_THRESHOLD


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Segment a long background recording into fixed-length "
        "_background_ clips and add them to training_v2."
    )
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="Report counts without writing files.")
    args = parser.parse_args()

    source_stem = os.path.splitext(os.path.basename(args.source))[0]

    print(f"Loading {args.source}...")
    audio = load_mono_16k(args.source)
    duration_sec = len(audio) / SAMPLE_RATE
    print(f"Loaded {duration_sec:.1f}s ({duration_sec / 60:.1f} min) at {SAMPLE_RATE} Hz")

    clips = segment(audio, OUTPUT_SEQUENCE_LENGTH)
    print(f"Segmented into {len(clips)} candidate {OUTPUT_SEQUENCE_LENGTH / SAMPLE_RATE:.2f}s windows")

    kept = [(i, clip) for i, clip in enumerate(clips) if not is_empty(clip)]
    dropped = len(clips) - len(kept)
    print(f"Kept {len(kept)} non-empty clips, dropped {dropped} as too quiet "
          f"(peak < {EMPTY_PEAK_THRESHOLD} or rms < {EMPTY_RMS_THRESHOLD})")

    rng = random.Random(args.seed)
    indices = list(range(len(kept)))
    rng.shuffle(indices)
    n_val = int(len(indices) * args.val_fraction)
    val_set = set(indices[:n_val])

    written = {"train": 0, "val": 0}
    manifest = {"source": os.path.relpath(args.source, ML_AUDIO_DIR), "clips": []}

    for shuffled_pos, (original_idx, clip) in enumerate(kept):
        split = "val" if shuffled_pos in val_set else "train"
        out_name = f"bgreal_{source_stem}__{split}__{written[split]:06d}.wav"
        out_dir = os.path.join(args.dataset_root, split, "_background_")
        out_path = os.path.join(out_dir, out_name)

        manifest["clips"].append({
            "split": split,
            "out_name": out_name,
            "source_window_index": original_idx,
        })

        if not args.dry_run:
            os.makedirs(out_dir, exist_ok=True)
            sf.write(out_path, clip, SAMPLE_RATE)
        written[split] += 1

    print(f"Train clips written: {written['train']}  Val clips written: {written['val']}")
    if args.dry_run:
        print("Dry run -- no files written.")

    os.makedirs(REPORT_DIR, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest_path = os.path.join(REPORT_DIR, f"background_segmentation_{timestamp}.json")
    manifest["generated_at"] = timestamp
    manifest["dry_run"] = args.dry_run
    manifest["total_candidate_windows"] = len(clips)
    manifest["dropped_empty"] = dropped
    manifest["written"] = written
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
