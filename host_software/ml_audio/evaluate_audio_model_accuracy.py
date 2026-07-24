import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import tensorflow as tf

from test_audio_command_classifier_microphone import (
    LABEL_NAMES,
    SAMPLE_RATE,
    CLIP_SECONDS,
    align_speech_to_fixed_length,
    waveform_to_spectrogram,
)


def predict_file(model, wav_path: Path):
    waveform, sr = sf.read(str(wav_path), dtype="float32")
    if sr != SAMPLE_RATE:
        raise ValueError(f"{wav_path} has sample rate {sr}, expected {SAMPLE_RATE}")

    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1)

    waveform = align_speech_to_fixed_length(waveform, target_samples=int(SAMPLE_RATE * CLIP_SECONDS))
    if waveform is None:
        return "hold", 1.0, np.zeros(len(LABEL_NAMES), dtype=np.float32)

    spec = waveform_to_spectrogram(waveform)
    logits = model(spec, training=False)
    probs = tf.nn.softmax(logits, axis=-1).numpy()[0]
    pred_id = int(np.argmax(probs))
    return LABEL_NAMES[pred_id], float(probs[pred_id]), probs


def gather_labeled_files(dataset_root: Path):
    labeled_files = []
    for label in LABEL_NAMES:
        class_dir = dataset_root / label
        if not class_dir.exists():
            continue
        for wav_file in sorted(class_dir.glob("*.wav")):
            labeled_files.append((label, wav_file))
    return labeled_files


def evaluate(model, dataset_root: Path):
    labeled_files = gather_labeled_files(dataset_root)
    if not labeled_files:
        raise FileNotFoundError(f"No .wav files found under {dataset_root}")

    confusion = defaultdict(lambda: defaultdict(int))
    detailed = []

    for expected, wav_path in labeled_files:
        predicted, confidence, _ = predict_file(model, wav_path)
        confusion[expected][predicted] += 1
        detailed.append((expected, predicted, confidence, wav_path.name))

    return detailed, confusion


def print_report(detailed, confusion):
    total = len(detailed)
    correct = sum(1 for expected, predicted, _, _ in detailed if expected == predicted)
    overall = correct / total if total else 0.0

    print("\n" + "=" * 72)
    print("Dataset Accuracy Report")
    print("=" * 72)
    print(f"Overall accuracy: {overall:.2%} ({correct}/{total})")

    print("\nPer-command accuracy:")
    for label in LABEL_NAMES:
        row_total = sum(confusion[label].values())
        row_correct = confusion[label].get(label, 0)
        row_acc = row_correct / row_total if row_total else 0.0
        print(f"  {label:10s}: {row_acc:.2%} ({row_correct}/{row_total})")

    print("\nConfusion counts (expected -> predicted):")
    for expected in LABEL_NAMES:
        line = ", ".join(f"{pred}:{confusion[expected].get(pred, 0)}" for pred in LABEL_NAMES)
        print(f"  {expected:10s} -> {line}")

    print("\nTop misclassifications:")
    mistakes = [row for row in detailed if row[0] != row[1]]
    if not mistakes:
        print("  None")
    else:
        for expected, predicted, confidence, name in mistakes[:20]:
            print(f"  {name}: expected={expected}, predicted={predicted}, conf={confidence:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate audio command model accuracy on labeled WAV files.")
    parser.add_argument(
        "--model",
        default=r"c:\Users\aritr\Downloads\ball_balance_video_controlled\host_software\ml_audio\models\audio_command_classifier\best_classifier.keras",
        help="Path to trained Keras model (.keras).",
    )
    parser.add_argument(
        "--dataset",
        default=r"c:\Users\aritr\Downloads\data\gold\audio\commands\test",
        help="Path to labeled dataset root with subfolders per command.",
    )
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)
    detailed, confusion = evaluate(model, Path(args.dataset))
    print_report(detailed, confusion)


if __name__ == "__main__":
    main()
