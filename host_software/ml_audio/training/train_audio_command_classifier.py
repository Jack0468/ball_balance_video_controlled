"""Train the audio command classifier end-to-end on the 12-class dataset.

Extracted from the ad hoc training loop that previously only existed inside
audio_command_classifier_aligned_before_deterministic_patch_final.ipynb (see
docs/plans/audio_eval_notebook_refactor_plan.md), and retargeted at the
actually-deployed 12-class layout instead of the notebook's original 6-class
one (the 12-class dataset/checkpoint were produced by different, unrecovered
code -- see the plan's "Discrepancy, now resolved" note).

IMPORTANT, corrects an assumption made earlier in this same effort: the
deployed AudioCommandClassifier (audio_command_classifier_pytorch.py)
registers its three conv/batchnorm blocks as buffers rather than
nn.Parameters, which looked like "a frozen feature extractor shared across
checkpoints, only the dense head varies." That is FALSE -- direct comparison
of models/pytorch_v3's state dict against that file's baked-in constants
shows every conv/batchnorm buffer differs substantially (e.g. norm_variance
off by 21.9, conv1_weight off by 0.54 max-abs), not just the dense head. So
the v3 checkpoint was produced by training/fine-tuning the *entire* network,
and a linear probe on the file's baked-in "frozen" features cannot reproduce
it (verified: applying v3's own dense head to features extracted from the
untouched baked-in conv/bn buffers gives ~10% accuracy, vs 86.3% for the real
v3 checkpoint). This script trains the whole network for real instead.

AudioCommandClassifier itself has no trainable-mode batchnorm (forward()
always calls F.batch_norm(..., training=False)), so it can't be trained
directly either -- it's an inference-only, already-exported shape. This
script defines a parallel trainable module with the identical architecture
(same channel counts, kernel sizes, resize-to-64x64, global-average-pool
head) using real nn.Conv2d/nn.BatchNorm2d/nn.Linear layers, trains it with
standard backprop, then exports the trained weights into
AudioCommandClassifier's buffer-keyed state-dict format so the result is a
drop-in-compatible checkpoint for evaluate_audio_classifier.py and
audio_receiver_pytorch.py -- no changes needed to either.

Usage:
    python train_audio_command_classifier.py
    python train_audio_command_classifier.py --dataset-root <path> --out-dir <path> --epochs 60
"""

import argparse
import json
import os
import sys
import time
from typing import Callable

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ML_AUDIO_DIR = os.path.dirname(SCRIPT_DIR)
HOST_SOFTWARE_DIR = os.path.dirname(ML_AUDIO_DIR)
if HOST_SOFTWARE_DIR not in sys.path:
    sys.path.insert(0, HOST_SOFTWARE_DIR)

from ml_audio.audio_dsp import (  # noqa: E402
    OUTPUT_SEQUENCE_LENGTH,
    SAMPLE_RATE,
    waveform_to_spectrogram,
)
from ml_audio.evaluations.evaluate_audio_classifier import (  # noqa: E402
    gather_labeled_files,
    pad_or_truncate,
)
from ml_audio.training.audio_augmentations import AugmentationConfig, PRESETS  # noqa: E402

DEFAULT_DATASET_ROOT = os.path.join(
    ML_AUDIO_DIR, "data", "synthetic+real_dataset_large", "training_v2"
)
DEFAULT_OUT_DIR = os.path.join(ML_AUDIO_DIR, "models", "pytorch_v4")
DEFAULT_CHECKPOINT_NAME = "audio_command_classifier_state_dict_v4.pth"

# Matches the eps values hardcoded in AudioCommandClassifier.__init__
# (NORMALIZATION_EPSILON / BATCH_NORMALIZATION_EPS / _1_EPS / _2_EPS in
# audio_command_classifier_pytorch.py) so the exported checkpoint's running
# stats are consistent with the eps used to produce them.
NORM_EPSILON = 1e-7
BN_EPSILON = 0.001
SPEC_SIZE = (64, 64)


def discover_labels(dataset_root: str) -> list[str]:
    """Class order = alphabetical scan of class folders -- the same convention
    labels.json already uses (see plan doc), not a hand-maintained list."""
    train_dir = os.path.join(dataset_root, "train")
    return sorted(
        name
        for name in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, name))
    )


def compute_resized_spectrogram(wav_path: str) -> torch.Tensor:
    """wav file -> log-magnitude spectrogram -> resized to the model's fixed
    64x64 input. Resizing is a deterministic, parameter-free step (bilinear
    interpolation), so it's safe to precompute and cache once rather than
    redo it every training epoch."""
    fitted = load_fitted_waveform(wav_path)
    return spectrogram_from_waveform(fitted)


def load_fitted_waveform(wav_path: str) -> np.ndarray:
    """wav file -> fixed-length raw waveform, no spectrogram yet. Split out
    from compute_resized_spectrogram so experiment_augmentations.py can
    apply waveform-level augmentation (noise-mix, speed-perturb, reverb --
    all only meaningful pre-STFT) before computing the spectrogram."""
    waveform, sr = sf.read(wav_path, dtype="float32")
    if sr != SAMPLE_RATE:
        raise ValueError(f"{wav_path} has sample rate {sr}, expected {SAMPLE_RATE}")
    return pad_or_truncate(waveform, OUTPUT_SEQUENCE_LENGTH)


def spectrogram_from_waveform(fitted_waveform: np.ndarray) -> torch.Tensor:
    spec = waveform_to_spectrogram(fitted_waveform)  # [1, frames, freq]
    spec = spec.unsqueeze(1)  # [1, 1, frames, freq]
    spec = F.interpolate(spec, size=SPEC_SIZE, mode="bilinear", align_corners=False)
    return spec[0, 0]  # [64, 64]


def build_waveform_cache(
    labeled_files: list[tuple[str, str]],
    label_to_idx: dict[str, int],
    log_every: int = 2000,
) -> tuple[np.ndarray, torch.Tensor]:
    """Cache fixed-length raw waveforms (not spectrograms) so augmentation
    -- which only makes sense pre-STFT -- can be applied fresh each epoch.
    Spectrograms are computed per-batch in train() instead of once here."""
    waveforms = np.empty((len(labeled_files), OUTPUT_SEQUENCE_LENGTH), dtype=np.float32)
    targets = torch.empty(len(labeled_files), dtype=torch.long)

    for i, (label, wav_path) in enumerate(labeled_files):
        waveforms[i] = load_fitted_waveform(wav_path)
        targets[i] = label_to_idx[label]
        if (i + 1) % log_every == 0:
            print(f"  waveforms: {i + 1}/{len(labeled_files)}")

    return waveforms, targets


def batch_spectrograms(
    waveforms: np.ndarray, device: torch.device | None = None, chunk_size: int = 512
) -> torch.Tensor:
    """Vectorized STFT, processed in chunks (optionally on `device`), instead
    of looping spectrogram_from_waveform() per-sample -- this model is tiny
    enough that a per-sample Python loop dominates wall-clock and leaves a
    GPU essentially idle regardless of training device. Chunked rather than
    one giant call so a large val/normalization-sample set doesn't risk
    blowing up GPU memory."""
    chunks = []
    for start in range(0, len(waveforms), chunk_size):
        wf = torch.as_tensor(waveforms[start:start + chunk_size], dtype=torch.float32)
        if device is not None:
            wf = wf.to(device)
        spec = waveform_to_spectrogram(wf)  # [b, frames, freq]
        spec = spec.unsqueeze(1)  # [b, 1, frames, freq]
        spec = F.interpolate(spec, size=SPEC_SIZE, mode="bilinear", align_corners=False)
        chunks.append(spec[:, 0])  # [b, 64, 64]
    return torch.cat(chunks, dim=0)


def compute_class_weights(train_targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    """sklearn-style 'balanced' weighting: n_samples / (n_classes * n_per_class).
    Motivated by the dataset audit finding a real 3.3x count gap between the
    original 6 classes (~2000-2200 train clips, ~70% real recordings) and the
    5 movement classes added later (~600 clips, 100% synthetic) -- see plan doc."""
    counts = torch.bincount(train_targets, minlength=num_classes).float()
    return counts.sum() / (num_classes * counts.clamp(min=1))


class TrainableAudioCommandClassifier(nn.Module):
    """Trainable mirror of AudioCommandClassifier's architecture (three
    conv/batchnorm/relu blocks, maxpool after the first two, global average
    pool, dense head) using real nn.Conv2d/nn.BatchNorm2d/nn.Linear layers so
    it can actually be backpropagated through -- AudioCommandClassifier
    itself is inference-only (buffers, not parameters; batch_norm always
    called with training=False)."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.norm_mean = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.norm_var = nn.Parameter(torch.ones(1), requires_grad=False)

        self.conv1 = nn.Conv2d(1, 12, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(12, eps=BN_EPSILON)
        self.conv2 = nn.Conv2d(12, 24, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(24, eps=BN_EPSILON)
        self.conv3 = nn.Conv2d(24, 48, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(48, eps=BN_EPSILON)
        self.dense = nn.Linear(48, num_classes)

    def set_input_normalization(self, mean: torch.Tensor, var: torch.Tensor) -> None:
        with torch.no_grad():
            self.norm_mean.copy_(mean)
            self.norm_var.copy_(var)

    def forward(self, spec_64x64: torch.Tensor) -> torch.Tensor:
        x = spec_64x64.unsqueeze(1)  # [B, 1, 64, 64]
        x = (x - self.norm_mean.view(1, -1, 1, 1)) / torch.sqrt(
            self.norm_var.view(1, -1, 1, 1) + NORM_EPSILON
        )

        # Conv -> ReLU -> BatchNorm, in that order -- matches
        # AudioCommandClassifier.forward() exactly (verified by round-trip
        # comparison; the more conventional Conv->BN->ReLU order produces a
        # checkpoint that's numerically wrong once BN diverges from its
        # near-identity initialization, since ReLU and BN don't commute).
        x = self.bn1(F.relu(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = self.bn2(F.relu(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = self.bn3(F.relu(self.conv3(x)))

        x = x.mean(dim=(2, 3))
        return self.dense(x)

    def to_inference_state_dict(self) -> dict:
        """Export in AudioCommandClassifier's buffer-keyed format -- a
        drop-in-compatible checkpoint for evaluate_audio_classifier.py /
        audio_receiver_pytorch.py, no changes needed there."""
        return {
            "norm_mean": self.norm_mean.detach().clone(),
            "norm_variance": self.norm_var.detach().clone(),
            "conv1_weight": self.conv1.weight.detach().clone(),
            "conv1_bias": self.conv1.bias.detach().clone(),
            "bn1_gamma": self.bn1.weight.detach().clone(),
            "bn1_beta": self.bn1.bias.detach().clone(),
            "bn1_mean": self.bn1.running_mean.detach().clone(),
            "bn1_var": self.bn1.running_var.detach().clone(),
            "conv2_weight": self.conv2.weight.detach().clone(),
            "conv2_bias": self.conv2.bias.detach().clone(),
            "bn2_gamma": self.bn2.weight.detach().clone(),
            "bn2_beta": self.bn2.bias.detach().clone(),
            "bn2_mean": self.bn2.running_mean.detach().clone(),
            "bn2_var": self.bn2.running_var.detach().clone(),
            "conv3_weight": self.conv3.weight.detach().clone(),
            "conv3_bias": self.conv3.bias.detach().clone(),
            "bn3_gamma": self.bn3.weight.detach().clone(),
            "bn3_beta": self.bn3.bias.detach().clone(),
            "bn3_mean": self.bn3.running_mean.detach().clone(),
            "bn3_var": self.bn3.running_var.detach().clone(),
            "dense_weight": self.dense.weight.detach().clone(),
            "dense_bias": self.dense.bias.detach().clone(),
        }


def train(
    model: TrainableAudioCommandClassifier,
    train_waveforms: np.ndarray,
    train_targets: torch.Tensor,
    val_specs: torch.Tensor,
    val_targets: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    augmentation: AugmentationConfig,
    noise_pool: np.ndarray,
    class_weights: torch.Tensor | None,
    patience: int | None,
    seed: int,
    device: torch.device,
    on_epoch_end: Callable[[int, float, bool, dict | None], None] | None = None,
) -> tuple[dict, float, list[float]]:
    """`on_epoch_end`, if given, is called after every epoch as
    (epoch, val_acc, is_best, best_state_or_None) -- best_state is only
    populated when is_best, so callers doing something expensive with it
    (e.g. writing a checkpoint to Drive) only pay that cost on improvement,
    not every epoch. Exists so long/unattended runs (a multi-seed Colab
    sweep left running overnight) can persist progress incrementally instead
    of only writing anything out after the entire run finishes -- a crash or
    disconnect partway through a run then only costs that run's progress
    since its own last improvement, not everything before it too."""
    model.to(device)
    val_specs = val_specs.to(device)
    val_targets = val_targets.to(device)
    if class_weights is not None:
        class_weights = class_weights.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)
    n = len(train_waveforms)

    best_val_acc = -1.0
    best_epoch = 0
    best_state = None
    val_accs: list[float] = []

    for epoch in range(1, epochs + 1):
        model.train()
        perm = np.random.RandomState(seed * 1000 + epoch).permutation(n)
        total_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            batch_waveforms = np.stack(
                [augmentation.apply(train_waveforms[i], OUTPUT_SEQUENCE_LENGTH, noise_pool, rng) for i in idx]
            )
            batch_specs = batch_spectrograms(batch_waveforms, device=device)
            batch_targets = train_targets[idx].to(device)

            optimizer.zero_grad()
            logits = model(batch_specs)
            loss = F.cross_entropy(logits, batch_targets, weight=class_weights)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)

        model.eval()
        with torch.no_grad():
            val_acc = (model(val_specs).argmax(dim=1) == val_targets).float().mean().item()
        val_accs.append(val_acc)

        print(f"  epoch {epoch:3d}/{epochs}  loss={total_loss / n:.4f}  val_acc={val_acc:.3%}")

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if on_epoch_end is not None:
            on_epoch_end(epoch, val_acc, is_best, best_state if is_best else None)

        if patience is not None and epoch - best_epoch >= patience:
            print(f"  early stopping: no val improvement in {patience} epochs (best epoch {best_epoch})")
            break

    return best_state, best_val_acc, val_accs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the audio command classifier end-to-end and export an "
        "AudioCommandClassifier-compatible checkpoint."
    )
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--checkpoint-name", default=DEFAULT_CHECKPOINT_NAME)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--augmentation", choices=sorted(PRESETS.keys()), default="none",
        help="Waveform augmentation preset (see audio_augmentations.py PRESETS). "
        "'none' reproduces the original v4/v5 training behavior exactly.",
    )
    parser.add_argument(
        "--class-weights", action="store_true",
        help="Weight the loss inversely by class frequency -- motivated by the "
        "3.3x count gap between the original 6 classes and the 5 movement "
        "classes added later (see plan doc).",
    )
    parser.add_argument(
        "--patience", type=int, default=None,
        help="Stop after this many epochs with no val-accuracy improvement. "
        "Default: no early stopping (train the full --epochs budget).",
    )
    parser.add_argument(
        "--device", default=None,
        help="torch device for training (e.g. 'cuda', 'cpu'). Default: cuda if "
        "available, else cpu. The exported checkpoint is always moved back to "
        "CPU before saving, regardless of training device.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")

    labels = discover_labels(args.dataset_root)
    label_to_idx = {label: i for i, label in enumerate(labels)}
    print(f"Discovered {len(labels)} classes: {labels}")

    train_files = gather_labeled_files(os.path.join(args.dataset_root, "train"), labels)
    val_files = gather_labeled_files(os.path.join(args.dataset_root, "val"), labels)
    print(f"train clips: {len(train_files)}  val clips: {len(val_files)}")

    t0 = time.time()
    print("Loading train waveforms...")
    train_waveforms, train_targets = build_waveform_cache(train_files, label_to_idx)
    print("Loading val waveforms + spectrograms (val is never augmented)...")
    val_waveforms, val_targets = build_waveform_cache(val_files, label_to_idx)
    val_specs = batch_spectrograms(val_waveforms, device=device)
    print(f"Waveform loading done in {time.time() - t0:.1f}s")

    # Input normalization stats from an unaugmented sample of train
    # spectrograms -- computing over the full set isn't necessary for two
    # scalars, and this keeps these stats identical regardless of which
    # augmentation preset is active.
    sample_specs = batch_spectrograms(train_waveforms[: min(4000, len(train_waveforms))])
    norm_mean = sample_specs.mean().reshape(1)
    norm_var = sample_specs.var(unbiased=False).reshape(1)
    print(f"Input normalization stats: mean={norm_mean.item():.4f} var={norm_var.item():.4f}")

    model = TrainableAudioCommandClassifier(num_classes=len(labels))
    model.set_input_normalization(norm_mean, norm_var)

    augmentation = PRESETS[args.augmentation]
    background_idx = label_to_idx.get("_background_")
    noise_pool = (
        train_waveforms[train_targets.numpy() == background_idx]
        if augmentation.use_noise and background_idx is not None
        else np.empty((0, OUTPUT_SEQUENCE_LENGTH), dtype=np.float32)
    )
    class_weights = compute_class_weights(train_targets, len(labels)) if args.class_weights else None
    print(f"Augmentation: {augmentation.name}  class_weights: {args.class_weights}  "
          f"weight_decay: {args.weight_decay}  patience: {args.patience}")

    print("Training...")
    t0 = time.time()
    best_state, best_val_acc, _ = train(
        model, train_waveforms, train_targets, val_specs, val_targets,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay, augmentation=augmentation,
        noise_pool=noise_pool, class_weights=class_weights,
        patience=args.patience, seed=args.seed, device=device,
    )
    print(f"Training done in {time.time() - t0:.1f}s. Best val accuracy: {best_val_acc:.3%}")

    # best_state is already CPU (train() clones checkpoints via .cpu()), so
    # the exported checkpoint is portable regardless of training device.
    model.to("cpu")
    model.load_state_dict(best_state)
    model.eval()

    os.makedirs(args.out_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.out_dir, args.checkpoint_name)
    torch.save(model.to_inference_state_dict(), checkpoint_path)
    labels_path = os.path.join(args.out_dir, "labels.json")
    with open(labels_path, "w") as f:
        json.dump(labels, f, indent=2)

    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved labels: {labels_path}")


if __name__ == "__main__":
    main()
