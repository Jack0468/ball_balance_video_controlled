"""Evaluate a trained audio command classifier and persist a confusion matrix.

Extracted from the ad hoc process that previously produced confusion matrices
as ephemeral notebook/stdout output only, never checked in (see
docs/plans/audio_eval_notebook_refactor_plan.md). Every run writes both the
raw counts (JSON) and a rendered plot (PNG) to reports/, so results are
reproducible and diffable across model or dataset changes instead of read
once off a screen.

IMPORTANT, confirmed by direct A/B testing against the known acc=0.870
baseline (docs/plans/audio_eval_notebook_refactor_plan.md): the live
inference path in audio_receiver_pytorch.py does NOT match how this model
was actually trained/validated, on four separate points:

  1. Label order. audio_receiver_pytorch.py hardcodes its own 12-class
     LABEL_NAMES order, which is NOT alphabetical. models/pytorch_v3/labels.json
     IS alphabetical (the standard convention for scanning class folders).
     Evaluating with the receiver's hardcoded order collapses accuracy to
     ~7% -- i.e. if that order is ever used to interpret a 12-class
     checkpoint's live predictions, commands are being decoded as the wrong
     words. This file uses labels.json's order, which is correct.
  2. align_speech_to_fixed_length's active-region crop (find peak, crop
     around it, pad) drops accuracy by ~7 points on already-isolated dataset
     clips -- it's tuned for finding speech inside a noisy rolling live
     buffer, not for clips that are already single, pre-cut utterances. This
     file just pads/truncates to the target length directly.
  3. Peak-renormalizing each clip to 0.95 costs another ~2-3 points here.
     Not applied.
  4. Applying the spectral-subtraction noise profile at all costs roughly
     30-40 points on this dataset -- it appears tuned for the live
     mic/robot-noise scenario, not clean/dataset audio. Off by default here;
     pass --apply-noise-profile to opt in for experimentation.

With all four reverted to the simpler/matching form, this script reproduces
~86.3% against the recorded 0.870 baseline (the ~0.7pt residual is fully
explained by the 5 val clips removed during the corruption-quarantine step).
This is a real, separate finding from the corruption/background-balance
work: the deployed receiver's preprocessing appears to diverge from what the
model was actually evaluated against, which is a plausible contributor to
the "incorrect outputs during concurrent operation" issue in its own right.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import soundfile as sf
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ML_AUDIO_DIR = os.path.dirname(SCRIPT_DIR)
HOST_SOFTWARE_DIR = os.path.dirname(ML_AUDIO_DIR)
if HOST_SOFTWARE_DIR not in sys.path:
    sys.path.insert(0, HOST_SOFTWARE_DIR)

from ml_audio.audio_command_classifier_pytorch import AudioCommandClassifier  # noqa: E402
from ml_audio.audio_receiver_pytorch import (  # noqa: E402
    OUTPUT_SEQUENCE_LENGTH,
    SAMPLE_RATE,
    waveform_to_spectrogram,
)

DEFAULT_MODEL_DIR = os.path.join(ML_AUDIO_DIR, "models", "pytorch_v3")
DEFAULT_CHECKPOINT = os.path.join(
    DEFAULT_MODEL_DIR, "audio_command_classifier_state_dict_v3.pth"
)
DEFAULT_LABELS = os.path.join(DEFAULT_MODEL_DIR, "labels.json")
DEFAULT_DATASET = os.path.join(
    ML_AUDIO_DIR, "data", "synthetic+real_dataset_large", "training_v2", "val"
)
DEFAULT_NOISE_PROFILE = os.path.join(ML_AUDIO_DIR, "models", "noise_profile.pt")
DEFAULT_REPORT_DIR = os.path.join(SCRIPT_DIR, "reports")


def load_labels(labels_path: str) -> list[str]:
    with open(labels_path) as f:
        return json.load(f)


def load_model(
    checkpoint_path: str, labels: list[str], device: torch.device
) -> AudioCommandClassifier:
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    num_classes = state_dict["dense_bias"].shape[0]
    if num_classes != len(labels):
        raise ValueError(
            f"Checkpoint '{checkpoint_path}' has {num_classes} output classes "
            f"but the labels list has {len(labels)} entries"
        )
    model = AudioCommandClassifier(num_classes=num_classes)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def pad_or_truncate(waveform: np.ndarray, target_samples: int) -> np.ndarray:
    """Fit a clip to the model's fixed input length with no other processing.

    Deliberately NOT align_speech_to_fixed_length: dataset clips are already
    single, pre-cut utterances, and that function's active-region crop (built
    for locating speech inside a noisy rolling live buffer) and peak
    renormalization measurably hurt accuracy here -- see module docstring.
    """
    audio = np.asarray(waveform, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if len(audio) > target_samples:
        audio = audio[:target_samples]
    if len(audio) < target_samples:
        audio = np.pad(audio, (0, target_samples - len(audio)))
    return audio.astype(np.float32)


def predict_clip(
    model: AudioCommandClassifier,
    wav_path: str,
    labels: list[str],
    noise_profile: torch.Tensor | None,
    device: torch.device,
) -> str:
    waveform, sr = sf.read(wav_path, dtype="float32")
    if sr != SAMPLE_RATE:
        raise ValueError(f"{wav_path} has sample rate {sr}, expected {SAMPLE_RATE}")

    fitted = pad_or_truncate(waveform, OUTPUT_SEQUENCE_LENGTH)
    spec = waveform_to_spectrogram(fitted, noise_profile=noise_profile).to(device)
    with torch.no_grad():
        logits = model(spec)
        probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()[0]
    return labels[int(np.argmax(probs))]


def gather_labeled_files(dataset_root: str, labels: list[str]) -> list[tuple[str, str]]:
    labeled_files = []
    for label in labels:
        class_dir = os.path.join(dataset_root, label)
        if not os.path.isdir(class_dir):
            continue
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith(".wav"):
                labeled_files.append((label, os.path.join(class_dir, fname)))
    return labeled_files


def run_evaluation(
    model: AudioCommandClassifier,
    dataset_root: str,
    labels: list[str],
    noise_profile: torch.Tensor | None,
    device: torch.device,
) -> tuple[np.ndarray, int]:
    labeled_files = gather_labeled_files(dataset_root, labels)
    if not labeled_files:
        raise FileNotFoundError(f"No .wav files found under {dataset_root}")

    label_to_idx = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)

    for true_label, wav_path in labeled_files:
        pred_label = predict_clip(model, wav_path, labels, noise_profile, device)
        matrix[label_to_idx[true_label], label_to_idx[pred_label]] += 1

    return matrix, len(labeled_files)


def save_plot(matrix: np.ndarray, labels: list[str], accuracy: float, out_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums != 0
    )

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(normalized, cmap="YlOrRd", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Confusion (row-normalised)  acc={accuracy:.3f}")

    for i in range(len(labels)):
        for j in range(len(labels)):
            count = matrix[i, j]
            if count > 0:
                ax.text(
                    j,
                    i,
                    str(count),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if normalized[i, j] > 0.5 else "black",
                )

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the PyTorch audio command classifier and persist a confusion matrix."
    )
    parser.add_argument("--model", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--labels", default=DEFAULT_LABELS)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--noise-profile", default=DEFAULT_NOISE_PROFILE)
    parser.add_argument(
        "--apply-noise-profile",
        action="store_true",
        help=(
            "Apply spectral-subtraction noise profile before classifying. "
            "Off by default: it costs ~30-40 accuracy points on this dataset "
            "(see module docstring) -- opt in only to reproduce/inspect that."
        ),
    )
    parser.add_argument("--out-dir", default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()

    device = torch.device("cpu")
    labels = load_labels(args.labels)
    model = load_model(args.model, labels, device)

    noise_profile = None
    if args.apply_noise_profile and os.path.exists(args.noise_profile):
        noise_profile = torch.load(
            args.noise_profile, map_location=device, weights_only=True
        )

    matrix, total = run_evaluation(model, args.dataset, labels, noise_profile, device)
    correct = int(np.trace(matrix))
    accuracy = correct / total if total else 0.0

    os.makedirs(args.out_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    json_path = os.path.join(args.out_dir, f"confusion_matrix_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(
            {
                "generated_at": timestamp,
                "model": os.path.relpath(args.model, ML_AUDIO_DIR),
                "dataset": os.path.relpath(args.dataset, ML_AUDIO_DIR),
                "noise_profile_applied": noise_profile is not None,
                "labels": labels,
                "accuracy": accuracy,
                "total_clips": total,
                "correct": correct,
                "matrix": matrix.tolist(),
            },
            f,
            indent=2,
        )

    png_path = os.path.join(args.out_dir, f"confusion_matrix_{timestamp}.png")
    save_plot(matrix, labels, accuracy, png_path)

    print(f"Accuracy: {accuracy:.3%} ({correct}/{total})")
    print(f"Raw counts + metadata: {json_path}")
    print(f"Plot: {png_path}")


if __name__ == "__main__":
    main()
