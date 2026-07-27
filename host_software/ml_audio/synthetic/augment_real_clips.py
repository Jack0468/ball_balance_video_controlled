#!/usr/bin/env python3
"""
Fold your real recordings into the synthetic corpus.

Takes your existing per-command WAV folders and produces many augmented
variants of each clip, using the SAME treatment as the synthetic generator:
random offset inside the 1.25 s window, background noise at random SNR,
random gain, slight speed change. Writes into the same train/val layout.

Input layout (either works):
    <in>/<command>/*.wav
    <in>/<speaker>/<command>/*.wav

Speakers are detected from the folder name or from the "speakerNN__" filename
prefix, and held out whole for validation -- no speaker leakage.

Run:
    python augment_real_clips.py --in .\\data\\silver\\audio\\commands \\
        --out .\\data\\synthetic --variants 20 --noise-dir .\\noise
"""

import argparse
import math
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SAMPLE_RATE = 16_000
CLIP_SECONDS = 1.25
TARGET_SAMPLES = int(SAMPLE_RATE * CLIP_SECONDS)

COMMANDS = ["go_blue", "go_green", "go_red", "go_yellow", "hold", "stop"]
SPEAKER_RE = re.compile(r"^(speaker[^_]*)__", re.IGNORECASE)


def to_16k(audio, src_rate):
    if src_rate == SAMPLE_RATE:
        return audio.astype(np.float32)
    g = math.gcd(int(src_rate), SAMPLE_RATE)
    return resample_poly(audio, SAMPLE_RATE // g, int(src_rate) // g).astype(np.float32)


def trim_silence(audio, thresh_ratio=0.02):
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak < 1e-6:
        return audio
    active = np.where(np.abs(audio) > peak * thresh_ratio)[0]
    return audio if len(active) == 0 else audio[active[0]:active[-1] + 1]


def synth_noise(rng, n):
    kind = rng.choice(["white", "pink", "brown", "hum"])
    if kind == "white":
        x = rng.standard_normal(n)
    elif kind == "pink":
        x = np.cumsum(rng.standard_normal(n)) / np.sqrt(np.arange(1, n + 1))
    elif kind == "brown":
        x = np.cumsum(rng.standard_normal(n))
    else:
        t = np.arange(n) / SAMPLE_RATE
        f = rng.uniform(48, 62)
        x = np.sin(2 * np.pi * f * t) + 0.4 * np.sin(4 * np.pi * f * t)
        x = x + 0.15 * rng.standard_normal(n)
    x = x - np.mean(x)
    peak = np.max(np.abs(x))
    return (x / peak).astype(np.float32) if peak > 1e-9 else x.astype(np.float32)


def load_noise_bank(noise_dir):
    if not noise_dir:
        return []
    bank = []
    for path in sorted(Path(noise_dir).rglob("*.wav")):
        try:
            audio, sr = sf.read(str(path), dtype="float32")
        except Exception:
            continue
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        audio = to_16k(audio, sr)
        if len(audio) >= TARGET_SAMPLES:
            bank.append(audio)
    return bank


def draw_noise(rng, bank, n):
    if bank and rng.random() < 0.7:
        clip = bank[rng.integers(len(bank))]
        start = int(rng.integers(0, len(clip) - n + 1))
        seg = clip[start:start + n].copy()
        peak = np.max(np.abs(seg))
        return seg / peak if peak > 1e-9 else seg
    return synth_noise(rng, n)


def speed_perturb(audio, rate, rng):
    """Resample-based speed change; also shifts pitch slightly, which is fine."""
    if abs(rate - 1.0) < 1e-3:
        return audio
    up, down = 1000, int(1000 * rate)
    g = math.gcd(up, down)
    return resample_poly(audio, up // g, down // g).astype(np.float32)


def make_variant(rng, speech, noise, snr_db, gain, speed):
    speech = speed_perturb(speech, speed, rng)
    window = np.zeros(TARGET_SAMPLES, dtype=np.float32)

    if len(speech) > TARGET_SAMPLES:
        start_in = int(rng.integers(0, len(speech) - TARGET_SAMPLES + 1))
        speech = speech[start_in:start_in + TARGET_SAMPLES]

    offset = int(rng.integers(0, TARGET_SAMPLES - len(speech) + 1))
    window[offset:offset + len(speech)] = speech

    s_rms = float(np.sqrt(np.mean(window ** 2)))
    n_rms = float(np.sqrt(np.mean(noise ** 2)))
    if s_rms > 1e-9 and n_rms > 1e-9:
        window = window + noise * ((s_rms / (10 ** (snr_db / 20.0))) / n_rms)

    window *= gain
    peak = float(np.max(np.abs(window)))
    if peak > 0.99:
        window = window / peak * 0.99
    return window.astype(np.float32)


def collect(in_root):
    """-> {command: [(speaker, path), ...]}"""
    in_root = Path(in_root)
    found = defaultdict(list)

    for command in COMMANDS:
        direct = in_root / command
        if direct.is_dir():
            for wav in sorted(direct.glob("*.wav")):
                m = SPEAKER_RE.match(wav.name)
                found[command].append((m.group(1).lower() if m else "unknown", wav))

    for speaker_dir in sorted(p for p in in_root.iterdir() if p.is_dir()):
        if speaker_dir.name in COMMANDS:
            continue
        for command in COMMANDS:
            sub = speaker_dir / command
            if sub.is_dir():
                for wav in sorted(sub.glob("*.wav")):
                    found[command].append((speaker_dir.name.lower(), wav))

    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_root", required=True,
                    help="Root of your existing recordings.")
    ap.add_argument("--out", default="data/synthetic",
                    help="Same --out you gave the synthetic generator.")
    ap.add_argument("--variants", type=int, default=20,
                    help="Augmented copies per source clip (default: 20).")
    ap.add_argument("--noise-dir", default=None)
    ap.add_argument("--val-speakers", type=int, default=1,
                    help="Speakers held out for validation (default: 1).")
    ap.add_argument("--snr-range", type=float, nargs=2, default=[5.0, 30.0])
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    found = collect(args.in_root)
    if not found:
        raise SystemExit(f"No per-command WAV folders found under {args.in_root}")

    speakers = sorted({spk for items in found.values() for spk, _ in items})
    print(f"speakers found: {speakers}")

    if len(speakers) > 1 and args.val_speakers > 0:
        held = set(random.sample(speakers, min(args.val_speakers, len(speakers) - 1)))
    else:
        held = set()
        print("Only one speaker -- splitting by clip instead (weaker validation).")
    print(f"validation speakers: {sorted(held) if held else '(none)'}")

    noise_bank = load_noise_bank(args.noise_dir)
    print(f"noise bank: {len(noise_bank)} files"
          + ("" if noise_bank else " (synthetic noise only)"))

    out_root = Path(args.out)
    totals = defaultdict(int)

    for command, items in sorted(found.items()):
        for idx, (speaker, wav_path) in enumerate(items):
            if held:
                split = "val" if speaker in held else "train"
            else:
                split = "val" if idx % 5 == 0 else "train"

            try:
                audio, sr = sf.read(str(wav_path), dtype="float32")
            except Exception as exc:
                print(f"  skip {wav_path.name}: {exc}")
                continue
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            speech = trim_silence(to_16k(audio, sr))
            if len(speech) < int(0.05 * SAMPLE_RATE):
                continue

            out_dir = out_root / split / command
            out_dir.mkdir(parents=True, exist_ok=True)

            n = args.variants if split == "train" else max(3, args.variants // 4)
            for v in range(n):
                clip = make_variant(
                    rng, speech,
                    draw_noise(rng, noise_bank, TARGET_SAMPLES),
                    snr_db=float(rng.uniform(*args.snr_range)),
                    gain=float(rng.uniform(0.25, 1.0)),
                    speed=float(rng.uniform(0.90, 1.10)),
                )
                name = f"real_{speaker}__{command}__{wav_path.stem}_{v:03d}.wav"
                sf.write(str(out_dir / name), clip, SAMPLE_RATE, subtype="PCM_16")
                totals[(split, command)] += 1

    print("\nWritten:")
    for (split, command), n in sorted(totals.items()):
        print(f"  {split}/{command}: {n}")
    print(f"\nMerged into {out_root.resolve()}")


if __name__ == "__main__":
    main()