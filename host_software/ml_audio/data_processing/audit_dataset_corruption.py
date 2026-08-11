"""Scan synthetic+real_dataset_large for empty and truncated command clips.

Extends the manual go_red audit in docs/dataset_info_audio.md to all 12 classes.
Reuses the same energy-gate thresholds as the production inference path
(audio_receiver_pytorch.align_speech_to_fixed_length) so "empty" here means
"the deployed model's own gate would already reject this clip as non-speech".
"""

import json
import os
from dataclasses import asdict, dataclass

import numpy as np
from scipy.io import wavfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ML_AUDIO_DIR = os.path.dirname(SCRIPT_DIR)
DATASET_ROOT = os.path.join(
    ML_AUDIO_DIR, "data", "synthetic+real_dataset_large", "training_v2"
)
REPORT_DIR = os.path.join(SCRIPT_DIR, "reports")

SPLITS = ["train", "val"]

# Same empty-clip gate as align_speech_to_fixed_length() in audio_receiver_pytorch.py.
EMPTY_PEAK_THRESHOLD = 0.03
EMPTY_RMS_THRESHOLD = 0.003

# A clip is flagged as a truncation *candidate* if its active-speech duration
# falls far below the rest of its own class. This is a statistical outlier
# check, not a transcript check -- it flags the same class of defect the
# manual audit found ("go", "go re", "red" are all short relative to a full
# "go_red" utterance) without needing ASR.
TRUNCATION_MAD_MULTIPLIER = 3.0
MIN_ACTIVE_DURATION_SEC = 0.12


@dataclass
class ClipStats:
    path: str
    split: str
    label: str
    duration_sec: float
    active_duration_sec: float
    peak: float
    rms: float
    is_empty: bool


def load_mono_float32(wav_path: str) -> tuple[np.ndarray, int]:
    sr, data = wavfile.read(wav_path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128.0) / 128.0
    else:
        data = data.astype(np.float32)

    if data.ndim > 1:
        data = np.mean(data, axis=1)
    return data, sr


def active_duration_sec(audio: np.ndarray, sr: int) -> float:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= 1e-6:
        return 0.0
    threshold = max(0.015, peak * 0.08)
    active = np.where(np.abs(audio) > threshold)[0]
    if len(active) == 0:
        return 0.0
    return float((active[-1] - active[0]) / sr)


def analyze_clip(wav_path: str, split: str, label: str) -> ClipStats:
    audio, sr = load_mono_float32(wav_path)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    rms = float(np.sqrt(np.mean(audio**2))) if audio.size else 0.0
    duration = len(audio) / sr if sr else 0.0
    active = active_duration_sec(audio, sr) if sr else 0.0
    is_empty = peak < EMPTY_PEAK_THRESHOLD or rms < EMPTY_RMS_THRESHOLD

    return ClipStats(
        path=os.path.relpath(wav_path, ML_AUDIO_DIR),
        split=split,
        label=label,
        duration_sec=duration,
        active_duration_sec=active,
        peak=peak,
        rms=rms,
        is_empty=is_empty,
    )


def scan_dataset() -> list[ClipStats]:
    results: list[ClipStats] = []
    for split in SPLITS:
        split_dir = os.path.join(DATASET_ROOT, split)
        if not os.path.isdir(split_dir):
            print(f"Warning: split dir not found: {split_dir}")
            continue
        labels = sorted(
            d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))
        )
        for label in labels:
            label_dir = os.path.join(split_dir, label)
            wav_files = sorted(f for f in os.listdir(label_dir) if f.lower().endswith(".wav"))
            print(f"Scanning {split}/{label}: {len(wav_files)} files...")
            for fname in wav_files:
                wav_path = os.path.join(label_dir, fname)
                try:
                    results.append(analyze_clip(wav_path, split, label))
                except Exception as e:
                    print(f"  Failed to read {wav_path}: {e}")
    return results


def flag_truncations(results: list[ClipStats]) -> set[str]:
    """Per (split, label) group, flag clips whose active duration is a low
    outlier relative to the group's median (robust z-score via MAD)."""
    flagged: set[str] = set()
    groups: dict[tuple[str, str], list[ClipStats]] = {}
    for r in results:
        if r.label == "_background_":
            continue  # background has no "full phrase" to truncate
        groups.setdefault((r.split, r.label), []).append(r)

    for (split, label), clips in groups.items():
        durations = np.array([c.active_duration_sec for c in clips if not c.is_empty])
        if len(durations) < 5:
            continue
        median = float(np.median(durations))
        mad = float(np.median(np.abs(durations - median))) or 1e-6
        for c in clips:
            if c.is_empty:
                continue
            robust_z = (median - c.active_duration_sec) / (1.4826 * mad)
            if (
                c.active_duration_sec < MIN_ACTIVE_DURATION_SEC
                or robust_z > TRUNCATION_MAD_MULTIPLIER
            ):
                flagged.add(c.path)
    return flagged


def main() -> None:
    os.makedirs(REPORT_DIR, exist_ok=True)
    results = scan_dataset()
    truncated_paths = flag_truncations(results)

    per_class_summary: dict[str, dict[str, int]] = {}
    for r in results:
        key = f"{r.split}/{r.label}"
        summary = per_class_summary.setdefault(
            key, {"total": 0, "empty": 0, "truncated": 0}
        )
        summary["total"] += 1
        if r.is_empty:
            summary["empty"] += 1
        if r.path in truncated_paths:
            summary["truncated"] += 1

    report = {
        "empty_thresholds": {
            "peak": EMPTY_PEAK_THRESHOLD,
            "rms": EMPTY_RMS_THRESHOLD,
        },
        "truncation_rule": {
            "mad_multiplier": TRUNCATION_MAD_MULTIPLIER,
            "min_active_duration_sec": MIN_ACTIVE_DURATION_SEC,
        },
        "per_class_summary": per_class_summary,
        "flagged_clips": [
            asdict(r)
            for r in results
            if r.is_empty or r.path in truncated_paths
        ],
    }

    out_path = os.path.join(REPORT_DIR, "dataset_corruption_audit.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    total_flagged = len(report["flagged_clips"])
    total_clips = len(results)
    print(f"\nScanned {total_clips} clips.")
    print(f"Flagged {total_flagged} clips (empty or truncated outlier).")
    print(f"Report written to {out_path}")
    print("\nPer-class summary (total / empty / truncated):")
    for key in sorted(per_class_summary):
        s = per_class_summary[key]
        print(f"  {key:24s} {s['total']:5d} / {s['empty']:4d} / {s['truncated']:4d}")


if __name__ == "__main__":
    main()
