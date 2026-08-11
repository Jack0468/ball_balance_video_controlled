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

import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ML_AUDIO_DIR = os.path.dirname(SCRIPT_DIR)
HOST_SOFTWARE_DIR = os.path.dirname(ML_AUDIO_DIR)
if HOST_SOFTWARE_DIR not in sys.path:
    sys.path.insert(0, HOST_SOFTWARE_DIR)

from ml_audio.audio_receiver_pytorch import (  # noqa: E402
    OUTPUT_SEQUENCE_LENGTH,
    SAMPLE_RATE,
    waveform_to_spectrogram,
)
from ml_audio.evaluations.evaluate_audio_classifier import (  # noqa: E402
    gather_labeled_files,
    pad_or_truncate,
)

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
    waveform, sr = sf.read(wav_path, dtype="float32")
    if sr != SAMPLE_RATE:
        raise ValueError(f"{wav_path} has sample rate {sr}, expected {SAMPLE_RATE}")
    fitted = pad_or_truncate(waveform, OUTPUT_SEQUENCE_LENGTH)
    spec = waveform_to_spectrogram(fitted)  # [1, frames, freq]
    spec = spec.unsqueeze(1)  # [1, 1, frames, freq]
    spec = F.interpolate(spec, size=SPEC_SIZE, mode="bilinear", align_corners=False)
    return spec[0, 0]  # [64, 64]


def build_spectrogram_cache(
    labeled_files: list[tuple[str, str]],
    label_to_idx: dict[str, int],
    log_every: int = 2000,
) -> tuple[torch.Tensor, torch.Tensor]:
    specs = torch.empty((len(labeled_files), *SPEC_SIZE), dtype=torch.float32)
    targets = torch.empty(len(labeled_files), dtype=torch.long)

    for i, (label, wav_path) in enumerate(labeled_files):
        specs[i] = compute_resized_spectrogram(wav_path)
        targets[i] = label_to_idx[label]
        if (i + 1) % log_every == 0:
            print(f"  spectrograms: {i + 1}/{len(labeled_files)}")

    return specs, targets


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
    train_specs: torch.Tensor,
    train_targets: torch.Tensor,
    val_specs: torch.Tensor,
    val_targets: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
) -> tuple[dict, float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n = train_specs.shape[0]

    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            batch_specs = train_specs[idx]
            batch_targets = train_targets[idx]

            optimizer.zero_grad()
            logits = model(batch_specs)
            loss = F.cross_entropy(logits, batch_targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)

        model.eval()
        with torch.no_grad():
            train_acc = (model(train_specs).argmax(dim=1) == train_targets).float().mean().item()
            val_acc = (model(val_specs).argmax(dim=1) == val_targets).float().mean().item()

        print(
            f"  epoch {epoch:3d}/{epochs}  loss={total_loss / n:.4f}  "
            f"train_acc={train_acc:.3%}  val_acc={val_acc:.3%}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    return best_state, best_val_acc


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
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    labels = discover_labels(args.dataset_root)
    label_to_idx = {label: i for i, label in enumerate(labels)}
    print(f"Discovered {len(labels)} classes: {labels}")

    train_files = gather_labeled_files(os.path.join(args.dataset_root, "train"), labels)
    val_files = gather_labeled_files(os.path.join(args.dataset_root, "val"), labels)
    print(f"train clips: {len(train_files)}  val clips: {len(val_files)}")

    t0 = time.time()
    print("Computing spectrograms for train split...")
    train_specs, train_targets = build_spectrogram_cache(train_files, label_to_idx)
    print("Computing spectrograms for val split...")
    val_specs, val_targets = build_spectrogram_cache(val_files, label_to_idx)
    print(f"Spectrogram caching done in {time.time() - t0:.1f}s")

    norm_mean = train_specs.mean().reshape(1)
    norm_var = train_specs.var(unbiased=False).reshape(1)
    print(f"Input normalization stats: mean={norm_mean.item():.4f} var={norm_var.item():.4f}")

    model = TrainableAudioCommandClassifier(num_classes=len(labels))
    model.set_input_normalization(norm_mean, norm_var)

    print("Training...")
    t0 = time.time()
    best_state, best_val_acc = train(
        model, train_specs, train_targets, val_specs, val_targets,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
    )
    print(f"Training done in {time.time() - t0:.1f}s. Best val accuracy: {best_val_acc:.3%}")

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
