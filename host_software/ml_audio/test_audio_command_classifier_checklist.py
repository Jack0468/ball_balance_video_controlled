import argparse
from collections import defaultdict

import numpy as np
import tensorflow as tf

from test_audio_command_classifier_microphone import (
    LABEL_NAMES,
    align_speech_to_fixed_length,
    record_audio,
    waveform_to_spectrogram,
)


def predict_once(model):
    waveform = record_audio()
    waveform = align_speech_to_fixed_length(waveform)
    if waveform is None:
        return None, None

    spec = waveform_to_spectrogram(waveform)
    logits = model(spec, training=False)
    probs = tf.nn.softmax(logits, axis=-1).numpy()[0]
    top_id = int(np.argmax(probs))
    return LABEL_NAMES[top_id], probs


def run_checklist(model, trials_per_command=5):
    results = []
    confusion = defaultdict(lambda: defaultdict(int))

    print("\nGuided microphone checklist")
    print("Commands:", ", ".join(LABEL_NAMES))
    print(f"Trials per command: {trials_per_command}\n")

    for expected in LABEL_NAMES:
        print("=" * 60)
        print(f"Expected command: {expected}")
        print("Speak the full phrase clearly after each prompt.")
        print("=" * 60)

        for trial in range(1, trials_per_command + 1):
            input(f"\nPress Enter for trial {trial}/{trials_per_command}...")
            predicted, probs = predict_once(model)

            if predicted is None:
                print("No speech detected. This trial is counted as 'hold'.")
                predicted = "hold"
                probs = np.zeros(len(LABEL_NAMES), dtype=np.float32)
                probs[LABEL_NAMES.index("hold")] = 1.0

            confidence = float(np.max(probs))
            confusion[expected][predicted] += 1
            results.append((expected, predicted, confidence))

            print(f"Predicted: {predicted} (confidence {confidence:.4f})")

    return results, confusion


def print_summary(results, confusion):
    print("\n" + "#" * 70)
    print("Accuracy summary")
    print("#" * 70)

    total = len(results)
    correct = sum(1 for expected, predicted, _ in results if expected == predicted)
    overall_acc = correct / total if total else 0.0
    print(f"Overall accuracy: {overall_acc:.2%} ({correct}/{total})")

    print("\nPer-command accuracy:")
    for label in LABEL_NAMES:
        row_total = sum(confusion[label].values())
        row_correct = confusion[label].get(label, 0)
        row_acc = row_correct / row_total if row_total else 0.0
        print(f"  {label:10s}: {row_acc:.2%} ({row_correct}/{row_total})")

    print("\nConfusion details (expected -> predicted counts):")
    for expected in LABEL_NAMES:
        counts = ", ".join(f"{pred}:{confusion[expected].get(pred, 0)}" for pred in LABEL_NAMES)
        print(f"  {expected:10s} -> {counts}")


def main():
    parser = argparse.ArgumentParser(description="Guided microphone checklist for all 6 audio commands.")
    parser.add_argument(
        "--model",
        default=r"c:\Users\aritr\Downloads\ball_balance_video_controlled\host_software\ml_audio\models\audio_command_classifier\best_classifier.keras",
        help="Path to trained Keras model (.keras).",
    )
    parser.add_argument("--trials", type=int, default=5, help="Trials per command.")
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)
    results, confusion = run_checklist(model, trials_per_command=args.trials)
    print_summary(results, confusion)


if __name__ == "__main__":
    main()
