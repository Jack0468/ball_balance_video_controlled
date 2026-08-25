"""Segment long raw bronze recordings (one file per label, several spoken
repetitions each) into fixed-length command clips and add them to
training_v2 as real recordings.

Built for `data/01_bronze_jack/` -- real human recordings of the 5 command
classes that were 100%-synthetic-TTS until now (`forward`, `backward`,
`left`, `right`, `go_grey`; see docs/plans/audio_eval_notebook_refactor_plan.md,
"Correction"/domain-gap sections). Filenames follow `<label>_<speaker>_<idx>.wav`
(e.g. `left_jack_0.wav`); label is everything before `_<speaker>_`.

Pipeline, mirroring the conventions already established elsewhere in this
module (audit_dataset_corruption.py's energy gate + MAD-based outlier check,
segment_background_recording.py's fixed-length clip format):

1. De-duplicate near-identical source files (same frame count + high
   waveform correlation) -- the raw batch had the same ~71s "forward"
   take saved three times (once stereo, twice mono/byte-identical).
2. Voice-activity segmentation: per-frame RMS-in-dB, Otsu-threshold
   (auto-adapts to each file's own recording level rather than a fixed
   dB cutoff), close short gaps (<250ms, avoids splitting a word at an
   internal dip), drop fragments (<150ms, breath/click noise), pad each
   surviving run with the same asymmetric margin
   align_speech_to_fixed_length() uses (80ms pre / 120ms post).
3. Per-label MAD-based duration outlier check (same 3.0 multiplier as
   audit_dataset_corruption.py) -- catches segmentation mistakes in
   either direction: too-short (a stray fragment that slipped past the
   150ms filter) or too-long (two repetitions merged into one because the
   pause between them was under 250ms).
4. Global gain correction: the raw bronze recordings turned out to be
   recorded at much lower input gain than the rest of the dataset (median
   segment peak ~0.045 vs. ~0.30 for the existing real_speakerNN clips --
   almost 7x quieter). Two risks from leaving that as-is: (a) the fixed
   empty-clip energy gate below, calibrated against the louder existing
   data, would wrongly flag genuine quiet-but-real speech as empty, and
   (b) more importantly, if every clip in 5 specific classes is
   systematically much quieter than every other class, the model could
   learn "quiet audio" as a shortcut for "movement command" that has
   nothing to do with the spoken content -- a real bias risk, not just a
   cosmetic mismatch. Fixed with ONE constant gain multiplier computed
   from the whole batch's median RMS against a same-sample of existing
   real clips, applied uniformly to every segment -- preserves each
   recording's natural internal dynamics (loud words still louder than
   quiet ones) while correcting the session-level gain, unlike the
   per-clip adaptive renormalization the live receiver already tried and
   found harmful (see Stage 2 in the plan doc) for a different reason
   (it also renormalizes noise/silence up to full-scale).
5. Fit to the fixed clip length (pad_or_truncate, 20000 samples / 1.25s --
   the format every other clip in training_v2 already uses) and re-run
   the standard empty-clip energy gate (peak<0.03 or rms<0.003) as a
   defense-in-depth check on top of the VAD.
6. Split accepted clips into train/val and write as
   `real_<speaker>__<label>__<split>__NNNNNN.wav` into
   `training_v2/{split}/<label>/` -- same naming convention the existing
   real_speakerNN clips already use.

Usage:
    python ingest_bronze_command_recordings.py
    python ingest_bronze_command_recordings.py --dry-run
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone

import numpy as np
import soundfile as sf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ML_AUDIO_DIR = os.path.dirname(SCRIPT_DIR)
HOST_SOFTWARE_DIR = os.path.dirname(ML_AUDIO_DIR)
if HOST_SOFTWARE_DIR not in sys.path:
    sys.path.insert(0, HOST_SOFTWARE_DIR)

from ml_audio.audio_dsp import OUTPUT_SEQUENCE_LENGTH, SAMPLE_RATE  # noqa: E402
from ml_audio.evaluations.evaluate_audio_classifier import pad_or_truncate  # noqa: E402

DEFAULT_SOURCE_DIR = os.path.join(ML_AUDIO_DIR, "data", "01_bronze_jack")
DEFAULT_SPEAKER = "jack"
DEFAULT_DATASET_ROOT = os.path.join(
    ML_AUDIO_DIR, "data", "synthetic+real_dataset_large", "training_v2"
)
REPORT_DIR = os.path.join(SCRIPT_DIR, "reports")

# Same empty-clip gate as align_speech_to_fixed_length() / audit_dataset_corruption.py.
EMPTY_PEAK_THRESHOLD = 0.03
EMPTY_RMS_THRESHOLD = 0.003
DURATION_MAD_MULTIPLIER = 3.0

FRAME_SAMPLES = 320   # 20ms @ 16kHz
HOP_SAMPLES = 160     # 10ms @ 16kHz
MIN_GAP_SEC = 0.25     # merge runs separated by less than this
MIN_RUN_SEC = 0.15     # drop runs shorter than this (breath/click noise)
MARGIN_PRE_SEC = 0.08
MARGIN_POST_SEC = 0.12


def load_mono(path: str) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        from scipy.signal import resample
        audio = resample(audio, int(len(audio) * SAMPLE_RATE / sr)).astype(np.float32)
    return audio


def dedupe_sources(paths: list[str]) -> tuple[list[str], list[dict]]:
    """Drop files that are near-identical to one already kept (same length,
    correlation > 0.999) -- catches accidental re-exports/re-saves of the
    same take, not just byte-identical duplicates."""
    kept: list[str] = []
    kept_audio: list[np.ndarray] = []
    dropped: list[dict] = []
    for path in paths:
        audio = load_mono(path)
        is_dupe = False
        for kept_path, kept_a in zip(kept, kept_audio):
            if len(audio) == len(kept_a):
                diff = np.abs(audio - kept_a).max()
                if diff < 1e-3:
                    dropped.append({"path": os.path.basename(path), "duplicate_of": os.path.basename(kept_path), "max_abs_diff": float(diff)})
                    is_dupe = True
                    break
        if not is_dupe:
            kept.append(path)
            kept_audio.append(audio)
    return kept, dropped


def otsu_threshold(values: np.ndarray) -> float:
    hist, edges = np.histogram(values, bins=256)
    hist = hist.astype(float)
    total = hist.sum()
    centers = (edges[:-1] + edges[1:]) / 2
    sum_all = np.sum(hist * centers)
    sum_bg = weight_bg = 0.0
    best_thresh, best_var = edges[0], -1.0
    for i in range(len(hist)):
        weight_bg += hist[i]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += hist[i] * centers[i]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_all - sum_bg) / weight_fg
        var_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if var_between > best_var:
            best_var = var_between
            best_thresh = centers[i]
    return best_thresh


def frame_rms_db(audio: np.ndarray) -> np.ndarray:
    n_frames = 1 + max(0, (len(audio) - FRAME_SAMPLES) // HOP_SAMPLES)
    rms = np.empty(n_frames)
    for i in range(n_frames):
        seg = audio[i * HOP_SAMPLES: i * HOP_SAMPLES + FRAME_SAMPLES]
        rms[i] = np.sqrt(np.mean(seg.astype(np.float64) ** 2) + 1e-12)
    return 20 * np.log10(rms + 1e-8)


def detect_segments(audio: np.ndarray) -> list[tuple[int, int]]:
    """Voice-activity segmentation -> list of (start_sample, end_sample)
    windows, each one repetition of the spoken word/phrase, with margin."""
    db = frame_rms_db(audio)
    active = db > otsu_threshold(db)

    min_gap_frames = int(MIN_GAP_SEC * SAMPLE_RATE / HOP_SAMPLES)
    closed = active.copy()
    i = 0
    while i < len(closed):
        if not closed[i]:
            j = i
            while j < len(closed) and not closed[j]:
                j += 1
            if (j - i) < min_gap_frames and 0 < i and j < len(closed):
                closed[i:j] = True
            i = j
        else:
            i += 1

    runs = []
    start = None
    for i, a in enumerate(closed):
        if a and start is None:
            start = i
        elif not a and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(closed)))

    segments = []
    for s, e in runs:
        if (e - s) * HOP_SAMPLES / SAMPLE_RATE < MIN_RUN_SEC:
            continue
        start_sample = max(0, s * HOP_SAMPLES - int(MARGIN_PRE_SEC * SAMPLE_RATE))
        end_sample = min(len(audio), e * HOP_SAMPLES + int(MARGIN_POST_SEC * SAMPLE_RATE))
        segments.append((start_sample, end_sample))
    return segments


def mad_outlier_mask(durations: list[float]) -> list[bool]:
    """True where a duration is a statistical outlier for its label (either
    direction) -- same method audit_dataset_corruption.py uses for
    truncation detection, applied here to catch merged/split segmentation
    mistakes instead."""
    arr = np.asarray(durations)
    median = np.median(arr)
    mad = np.median(np.abs(arr - median)) + 1e-8
    z = 0.6745 * (arr - median) / mad
    return list(np.abs(z) > DURATION_MAD_MULTIPLIER)


def reference_target_rms(dataset_root: str, sample_n: int = 60) -> float:
    """Median RMS of a sample of existing real_speakerNN clips, as the gain
    target -- so the new recordings land at a level consistent with the
    rest of the dataset rather than an arbitrary fixed number."""
    import glob
    candidates = glob.glob(os.path.join(dataset_root, "train", "*", "real_speaker*.wav"))
    if not candidates:
        return 0.05  # fallback if no reference clips are found
    rng = random.Random(0)
    sample = rng.sample(candidates, min(sample_n, len(candidates)))
    rms_values = []
    for path in sample:
        audio, _ = sf.read(path, dtype="float32")
        rms_values.append(np.sqrt(np.mean(audio ** 2)))
    return float(np.median(rms_values))


def compute_gain_factor(all_raw_segments: list[np.ndarray], target_rms: float) -> float:
    """One constant multiplier for the whole batch (not per-clip) -- fixes
    the session-level recording gain while preserving each clip's natural
    internal dynamics. Clamped so the loudest segment in the batch doesn't
    clip after scaling."""
    rms_values = np.array([np.sqrt(np.mean(seg ** 2)) for seg in all_raw_segments])
    peak_values = np.array([np.max(np.abs(seg)) for seg in all_raw_segments])
    desired_gain = target_rms / (np.median(rms_values) + 1e-8)
    max_safe_gain = 0.98 / (np.max(peak_values) + 1e-8)
    return float(min(desired_gain, max_safe_gain))


def is_empty_clip(clip: np.ndarray) -> bool:
    peak = np.max(np.abs(clip))
    rms = np.sqrt(np.mean(clip ** 2))
    return peak < EMPTY_PEAK_THRESHOLD or rms < EMPTY_RMS_THRESHOLD


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Segment bronze command recordings into training_v2 clips."
    )
    parser.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--speaker", default=DEFAULT_SPEAKER)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="Report without writing files.")
    args = parser.parse_args()

    source_files = sorted(
        os.path.join(args.source_dir, f)
        for f in os.listdir(args.source_dir)
        if f.lower().endswith(".wav")
    )
    print(f"Found {len(source_files)} source files in {args.source_dir}")

    by_label: dict[str, list[str]] = {}
    for path in source_files:
        fname = os.path.basename(path)
        marker = f"_{args.speaker}_"
        if marker not in fname:
            print(f"  WARNING: {fname} doesn't match '<label>{marker}<idx>.wav', skipping")
            continue
        label = fname.split(marker)[0]
        by_label.setdefault(label, []).append(path)

    manifest = {"speaker": args.speaker, "labels": {}}
    rng = random.Random(args.seed)
    totals = {"raw_segments": 0, "mad_outliers": 0, "empty_after_pad": 0, "accepted": 0}

    # Pass 1: detect every segment across all labels first, so the gain
    # correction is computed from the whole batch (one consistent session
    # gain), not recomputed per label.
    per_label_segments: dict[str, list[tuple[str, np.ndarray]]] = {}
    per_label_dedup: dict[str, list[dict]] = {}
    all_raw_clips: list[np.ndarray] = []
    for label in sorted(by_label):
        kept_paths, dropped = dedupe_sources(by_label[label])
        per_label_dedup[label] = dropped
        if dropped:
            print(f"[{label}] deduped {len(dropped)} near-identical file(s): "
                  f"{[d['path'] + ' == ' + d['duplicate_of'] for d in dropped]}")
        segments = []
        for path in kept_paths:
            audio = load_mono(path)
            for start, end in detect_segments(audio):
                clip = audio[start:end]
                segments.append((os.path.basename(path), clip))
                all_raw_clips.append(clip)
        per_label_segments[label] = segments

    target_rms = reference_target_rms(args.dataset_root)
    gain = compute_gain_factor(all_raw_clips, target_rms)
    print(f"\nGain correction: target_rms={target_rms:.4f} (from existing real_speaker clips) "
          f"-> applying constant {gain:.2f}x to the whole batch\n")

    # Pass 2: apply gain, filter, split, write.
    for label in sorted(by_label):
        all_segments = per_label_segments[label]
        dropped = per_label_dedup[label]
        raw_durations = [(len(clip) / SAMPLE_RATE) for _, clip in all_segments]

        outlier_mask = mad_outlier_mask(raw_durations) if len(raw_durations) > 1 else [False] * len(raw_durations)

        accepted_clips = []
        excluded = {"mad_outlier": [], "empty_after_pad": []}
        for (src, raw_clip), dur, is_outlier in zip(all_segments, raw_durations, outlier_mask):
            if is_outlier:
                excluded["mad_outlier"].append({"source": src, "duration_sec": round(dur, 3)})
                continue
            gained = np.clip(raw_clip * gain, -1.0, 1.0).astype(np.float32)
            fitted = pad_or_truncate(gained, OUTPUT_SEQUENCE_LENGTH)
            if is_empty_clip(fitted):
                excluded["empty_after_pad"].append({"source": src, "duration_sec": round(dur, 3)})
                continue
            accepted_clips.append(fitted)

        totals["raw_segments"] += len(all_segments)
        totals["mad_outliers"] += len(excluded["mad_outlier"])
        totals["empty_after_pad"] += len(excluded["empty_after_pad"])
        totals["accepted"] += len(accepted_clips)

        indices = list(range(len(accepted_clips)))
        rng.shuffle(indices)
        n_val = max(1, int(len(indices) * args.val_fraction)) if indices else 0
        val_set = set(indices[:n_val])

        written = {"train": 0, "val": 0}
        for shuffled_pos, idx in enumerate(indices):
            split = "val" if shuffled_pos in val_set else "train"
            out_name = f"real_{args.speaker}__{label}__{split}__{written[split]:06d}.wav"
            out_dir = os.path.join(args.dataset_root, split, label)
            if not args.dry_run:
                os.makedirs(out_dir, exist_ok=True)
                sf.write(os.path.join(out_dir, out_name), accepted_clips[idx], SAMPLE_RATE)
            written[split] += 1

        print(f"[{label}] raw_segments={len(all_segments)} "
              f"mad_outliers={len(excluded['mad_outlier'])} empty_after_pad={len(excluded['empty_after_pad'])} "
              f"accepted={len(accepted_clips)}  -> train={written['train']} val={written['val']}")
        if excluded["mad_outlier"]:
            print(f"    excluded as duration outliers: {excluded['mad_outlier']}")
        if excluded["empty_after_pad"]:
            print(f"    excluded as empty after padding: {excluded['empty_after_pad']}")

        manifest["labels"][label] = {
            "source_files": [os.path.basename(p) for p in by_label[label]],
            "deduped_out": dropped,
            "raw_segments": len(all_segments),
            "excluded": excluded,
            "accepted": len(accepted_clips),
            "written": written,
        }

    print(f"\nTotals: raw_segments={totals['raw_segments']} "
          f"mad_outliers={totals['mad_outliers']} empty_after_pad={totals['empty_after_pad']} "
          f"accepted={totals['accepted']}")
    if args.dry_run:
        print("Dry run -- no files written.")

    os.makedirs(REPORT_DIR, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest_path = os.path.join(REPORT_DIR, f"bronze_ingest_{timestamp}.json")
    manifest["generated_at"] = timestamp
    manifest["dry_run"] = args.dry_run
    manifest["totals"] = totals
    manifest["gain_correction"] = {"target_rms": target_rms, "applied_gain": gain}
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
