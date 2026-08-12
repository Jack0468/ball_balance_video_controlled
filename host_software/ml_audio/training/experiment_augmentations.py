"""Local ablation: which waveform augmentations actually help, before
spending Colab GPU time on the full tuned/multi-seed retrain.

Motivation (see docs/plans/audio_eval_notebook_refactor_plan.md): v5's
background diversification improved every offline metric but regressed the
live-stream test, and a closer look at the dataset found the real issue
isn't just noise robustness -- 6 of the 12 classes (the 5 movement commands
plus effectively most of _background_) are ~100% synthetic-TTS with zero
real human recordings, unlike the original 6 classes which are ~70% real.
Augmentation (noise-mixing, speed perturbation, synthetic reverb) is the
cheap way to fake some of that missing real-world variation. But we don't
know yet which of these techniques actually move validation accuracy versus
just adding noise (in the statistical sense) to training -- rather than
guess, or spend a full Colab multi-seed sweep finding out, this runs short
(default 25-epoch), same-seed, single-technique-at-a-time comparisons on a
stratified subset of the training set, fast enough to iterate on a CPU
laptop. Winner(s) here become the augmentation config for the real Colab
tuning pass, not a final model -- nothing produced by this script is meant
to be deployed.

Usage:
    python experiment_augmentations.py
    python experiment_augmentations.py --epochs 20 --train-fraction 0.25
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ML_AUDIO_DIR = os.path.dirname(SCRIPT_DIR)
HOST_SOFTWARE_DIR = os.path.dirname(ML_AUDIO_DIR)
if HOST_SOFTWARE_DIR not in sys.path:
    sys.path.insert(0, HOST_SOFTWARE_DIR)

from ml_audio.evaluations.evaluate_audio_classifier import gather_labeled_files  # noqa: E402
from ml_audio.training.audio_augmentations import AugmentationConfig  # noqa: E402
from ml_audio.training.train_audio_command_classifier import (  # noqa: E402
    DEFAULT_DATASET_ROOT,
    TrainableAudioCommandClassifier,
    discover_labels,
    load_fitted_waveform,
    spectrogram_from_waveform,
)

REPORT_DIR = os.path.join(SCRIPT_DIR, "reports")

CONFIGS = [
    AugmentationConfig("baseline"),
    AugmentationConfig("noise_mix", use_noise=True),
    AugmentationConfig("speed_perturb", use_speed=True),
    AugmentationConfig("reverb", use_reverb=True),
    AugmentationConfig("light_gain_shift", use_gain=True, use_shift=True),
    AugmentationConfig(
        "combined", use_noise=True, use_speed=True, use_reverb=True,
        use_gain=True, use_shift=True, prob=0.5,
    ),
    # Follow-up after the first ablation pass: reverb was a clear, large
    # regression (best_val_acc 57.3% vs. 76.5% baseline); this config checks
    # whether dropping just reverb turns "combined" into a real winner
    # rather than a wash.
    AugmentationConfig(
        "combined_no_reverb", use_noise=True, use_speed=True,
        use_gain=True, use_shift=True, prob=0.5,
    ),
]


def stratified_subsample(
    labeled_files: list[tuple[str, str]], fraction: float, seed: int
) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    by_label: dict[str, list[tuple[str, str]]] = {}
    for label, path in labeled_files:
        by_label.setdefault(label, []).append((label, path))
    subsample = []
    for label, items in by_label.items():
        rng.shuffle(items)
        n = max(1, int(len(items) * fraction))
        subsample.extend(items[:n])
    return subsample


def load_waveform_cache(labeled_files, label_to_idx, log_every=1000):
    n = len(labeled_files)
    waveforms = np.empty((n, 20000), dtype=np.float32)
    targets = torch.empty(n, dtype=torch.long)
    for i, (label, wav_path) in enumerate(labeled_files):
        waveforms[i] = load_fitted_waveform(wav_path)
        targets[i] = label_to_idx[label]
        if (i + 1) % log_every == 0:
            print(f"    waveforms: {i + 1}/{n}")
    return waveforms, targets


def batch_spectrograms(waveforms: np.ndarray) -> torch.Tensor:
    specs = torch.empty((len(waveforms), 64, 64), dtype=torch.float32)
    for i, w in enumerate(waveforms):
        specs[i] = spectrogram_from_waveform(w)
    return specs


def run_config(
    config: AugmentationConfig,
    train_waveforms: np.ndarray,
    train_targets: torch.Tensor,
    val_specs: torch.Tensor,
    val_targets: torch.Tensor,
    noise_pool: np.ndarray,
    num_classes: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> tuple[float, list[float]]:
    torch.manual_seed(seed)
    model = TrainableAudioCommandClassifier(num_classes=num_classes)

    # Input normalization stats from an unaugmented pass over the training
    # waveforms -- keeping this fixed across configs isolates the
    # augmentation's effect from incidental shifts in these two scalars.
    with torch.no_grad():
        sample_specs = batch_spectrograms(train_waveforms[: min(2000, len(train_waveforms))])
        model.set_input_normalization(sample_specs.mean().reshape(1), sample_specs.var(unbiased=False).reshape(1))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    n = len(train_waveforms)
    val_accs = []

    for epoch in range(1, epochs + 1):
        model.train()
        perm = np.random.RandomState(seed * 1000 + epoch).permutation(n)
        total_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            batch_waveforms = np.stack(
                [config.apply(train_waveforms[i], 20000, noise_pool, rng) for i in idx]
            )
            batch_specs = batch_spectrograms(batch_waveforms)
            batch_targets = train_targets[idx]

            optimizer.zero_grad()
            logits = model(batch_specs)
            loss = F.cross_entropy(logits, batch_targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)

        model.eval()
        with torch.no_grad():
            val_acc = (model(val_specs).argmax(dim=1) == val_targets).float().mean().item()
        val_accs.append(val_acc)
        print(f"    [{config.name}] epoch {epoch:2d}/{epochs}  loss={total_loss / n:.4f}  val_acc={val_acc:.3%}")

    return max(val_accs), val_accs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablate augmentation techniques on a subset of the training set."
    )
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--train-fraction", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--configs", default=None,
        help="Comma-separated subset of config names to run (default: all).",
    )
    args = parser.parse_args()

    configs_to_run = CONFIGS
    if args.configs:
        wanted = set(args.configs.split(","))
        configs_to_run = [c for c in CONFIGS if c.name in wanted]

    labels = discover_labels(args.dataset_root)
    label_to_idx = {label: i for i, label in enumerate(labels)}
    print(f"Classes: {labels}")

    train_files_full = gather_labeled_files(os.path.join(args.dataset_root, "train"), labels)
    val_files = gather_labeled_files(os.path.join(args.dataset_root, "val"), labels)
    train_files = stratified_subsample(train_files_full, args.train_fraction, args.seed)
    print(f"train subset: {len(train_files)}/{len(train_files_full)}  val: {len(val_files)}")

    t0 = time.time()
    print("Loading train waveforms...")
    train_waveforms, train_targets = load_waveform_cache(train_files, label_to_idx)
    print("Loading val waveforms + spectrograms (no augmentation applied to val, ever)...")
    val_waveforms, val_targets = load_waveform_cache(val_files, label_to_idx)
    val_specs = batch_spectrograms(val_waveforms)
    print(f"Waveform loading done in {time.time() - t0:.1f}s")

    background_idx = label_to_idx["_background_"]
    noise_pool = train_waveforms[train_targets.numpy() == background_idx]
    print(f"Noise pool: {len(noise_pool)} background clips")

    results = {}
    for config in configs_to_run:
        print(f"\n=== Config: {config.name} ===")
        t0 = time.time()
        best_acc, val_accs = run_config(
            config, train_waveforms, train_targets, val_specs, val_targets,
            noise_pool, num_classes=len(labels), epochs=args.epochs,
            batch_size=args.batch_size, lr=args.lr, seed=args.seed,
        )
        elapsed = time.time() - t0
        print(f"=== {config.name}: best val_acc={best_acc:.3%} ({elapsed:.0f}s) ===")
        results[config.name] = {"best_val_acc": best_acc, "val_accs": val_accs, "seconds": elapsed}

    print("\n--- Summary ---")
    for name, r in sorted(results.items(), key=lambda kv: -kv[1]["best_val_acc"]):
        print(f"  {name:18s} best_val_acc={r['best_val_acc']:.3%}")

    os.makedirs(REPORT_DIR, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = os.path.join(REPORT_DIR, f"augmentation_ablation_{timestamp}.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "generated_at": timestamp,
                "train_fraction": args.train_fraction,
                "epochs": args.epochs,
                "train_subset_size": len(train_files),
                "val_size": len(val_files),
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nReport: {out_path}")


if __name__ == "__main__":
    main()
