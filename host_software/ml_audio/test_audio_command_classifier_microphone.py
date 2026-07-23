import argparse
import numpy as np
import sounddevice as sd
import tensorflow as tf

SAMPLE_RATE = 16_000
CLIP_SECONDS = 4
# Model was trained on short utterances; keep analysis window short for sequence decoding.
MODEL_WINDOW_SECONDS = 1.25
OUTPUT_SEQUENCE_LENGTH = int(SAMPLE_RATE * MODEL_WINDOW_SECONDS)
N_FFT = 255
HOP_LENGTH = 128

LABEL_NAMES = ["go_blue", "go_green", "go_red", "go_yellow", "hold", "stop"]


def align_speech_to_fixed_length(audio, target_samples=OUTPUT_SEQUENCE_LENGTH):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    peak = np.max(np.abs(audio))
    rms = np.sqrt(np.mean(audio ** 2))

    if peak < 0.03 or rms < 0.003:
        return None

    threshold = max(0.015, peak * 0.08)
    active = np.where(np.abs(audio) > threshold)[0]
    if len(active) == 0:
        return None

    start = max(0, active[0] - int(0.08 * SAMPLE_RATE))
    end = min(len(audio), active[-1] + int(0.12 * SAMPLE_RATE))
    audio = audio[start:end]

    if len(audio) > target_samples:
        audio = audio[:target_samples]
    if len(audio) < target_samples:
        audio = np.pad(audio, (0, target_samples - len(audio)))

    peak = np.max(np.abs(audio))
    if peak > 1e-6:
        audio = audio / peak * 0.95

    return audio.astype(np.float32)


def waveform_to_spectrogram(waveform):
    waveform_tf = tf.convert_to_tensor(waveform[np.newaxis, :], dtype=tf.float32)
    spec = tf.signal.stft(waveform_tf, frame_length=N_FFT, frame_step=HOP_LENGTH)
    spec = tf.abs(spec)
    spec = tf.math.log(spec + 1e-6)
    spec = spec[..., tf.newaxis]
    return spec


def record_audio(duration=CLIP_SECONDS):
    print(f"Recording {duration:.2f} seconds from your default microphone...")
    recording = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    waveform = np.squeeze(recording, axis=-1)
    return waveform


def predict_from_microphone(model, analysis_window_seconds=MODEL_WINDOW_SECONDS):
    waveform = record_audio(CLIP_SECONDS)
    target_samples = int(SAMPLE_RATE * analysis_window_seconds)
    waveform = align_speech_to_fixed_length(waveform, target_samples=target_samples)
    if waveform is None:
        print("No speech detected. Please speak louder and try again.")
        return

    spec = waveform_to_spectrogram(waveform)
    logits = model(spec, training=False)
    probs = tf.nn.softmax(logits, axis=-1).numpy()[0]

    top_id = int(np.argmax(probs))
    print("\nPrediction:")
    for i, label in enumerate(LABEL_NAMES):
        print(f"  {label}: {probs[i]:.4f}")
    print(f"\nTop command: {LABEL_NAMES[top_id]} (confidence {probs[top_id]:.4f})")


def collapse_sequence_events(events, merge_gap_seconds=0.6):
    if not events:
        return []

    merged = []
    for event in events:
        if not merged:
            merged.append(
                {
                    "label": event["label"],
                    "start_time": event["time"],
                    "end_time": event["time"],
                    "best_conf": event["conf"],
                    "count": 1,
                }
            )
            continue

        last = merged[-1]
        same_label = event["label"] == last["label"]
        close_enough = (event["time"] - last["end_time"]) <= merge_gap_seconds

        if same_label and close_enough:
            last["end_time"] = event["time"]
            last["best_conf"] = max(last["best_conf"], event["conf"])
            last["count"] += 1
        else:
            merged.append(
                {
                    "label": event["label"],
                    "start_time": event["time"],
                    "end_time": event["time"],
                    "best_conf": event["conf"],
                    "count": 1,
                }
            )

    return merged


def predict_sequence_from_microphone(
    model,
    total_duration,
    analysis_window_seconds,
    step_seconds,
    min_confidence,
    min_margin,
    min_event_windows,
    merge_gap_seconds,
    include_hold,
    include_stop,
    max_commands,
):
    waveform = record_audio(total_duration)

    window_samples = int(SAMPLE_RATE * analysis_window_seconds)
    step_samples = max(1, int(step_seconds * SAMPLE_RATE))

    if len(waveform) < window_samples:
        print("Recording is too short for one analysis window.")
        return

    raw_events = []
    for start in range(0, len(waveform) - window_samples + 1, step_samples):
        chunk = waveform[start : start + window_samples]
        aligned = align_speech_to_fixed_length(chunk, target_samples=window_samples)
        if aligned is None:
            continue

        spec = waveform_to_spectrogram(aligned)
        logits = model(spec, training=False)
        probs = tf.nn.softmax(logits, axis=-1).numpy()[0]

        top_id = int(np.argmax(probs))
        top_label = LABEL_NAMES[top_id]
        top_conf = float(probs[top_id])
        top_two = np.partition(probs, -2)[-2:]
        margin = float(top_two[-1] - top_two[-2])

        if top_conf < min_confidence:
            continue
        if margin < min_margin:
            continue
        if not include_hold and top_label == "hold":
            continue
        if not include_stop and top_label == "stop":
            continue

        # Use window midpoint as event time on the full recording timeline.
        event_time = (start + window_samples / 2) / SAMPLE_RATE
        raw_events.append({"time": event_time, "label": top_label, "conf": top_conf})

    merged_events = collapse_sequence_events(raw_events, merge_gap_seconds=merge_gap_seconds)
    merged_events = [event for event in merged_events if event["count"] >= min_event_windows]

    if max_commands > 0:
        capped_events = []
        seen_labels = []
        for event in merged_events:
            if not seen_labels:
                seen_labels.append(event["label"])
                capped_events.append(event)
                continue

            if event["label"] == seen_labels[-1]:
                capped_events[-1] = event
                continue

            if event["label"] in seen_labels:
                # Ignore command re-entries once we have moved on to reduce transition noise.
                continue

            if len(seen_labels) < max_commands:
                seen_labels.append(event["label"])
                capped_events.append(event)

        merged_events = capped_events

    if not merged_events:
        print("\nNo confident command sequence detected.")
        return

    print("\nDetected timeline:")
    for event in merged_events:
        if abs(event["start_time"] - event["end_time"]) < 1e-6:
            time_part = f"{event['start_time']:.2f}s"
        else:
            time_part = f"{event['start_time']:.2f}s-{event['end_time']:.2f}s"
        print(
            f"  {time_part}: {event['label']} "
            f"(best conf {event['best_conf']:.3f}, windows {event['count']})"
        )

    command_chain = " -> ".join(event["label"] for event in merged_events)
    print(f"\nSequence: {command_chain}")


def main():
    parser = argparse.ArgumentParser(description="Test the trained Keras audio classifier with microphone input.")
    parser.add_argument(
        "--model",
        default=r"c:\Users\aritr\Downloads\ball_balance_video_controlled\host_software\ml_audio\models\audio_command_classifier\best_classifier.keras",
        help="Path to trained Keras model (.keras).",
    )
    parser.add_argument(
        "--mode",
        choices=["single", "sequence"],
        default="single",
        help="single: one command in one clip. sequence: multiple commands in a longer recording.",
    )
    parser.add_argument(
        "--sequence-duration",
        type=float,
        default=4.0,
        help="Recording length in seconds for sequence mode.",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=MODEL_WINDOW_SECONDS,
        help="Analysis window length in seconds. Smaller values are better for detecting command sequences.",
    )
    parser.add_argument(
        "--step-seconds",
        type=float,
        default=0.2,
        help="Sliding-window step size in seconds for sequence mode.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.6,
        help="Minimum confidence required to keep a detected command in sequence mode.",
    )
    parser.add_argument(
        "--min-margin",
        type=float,
        default=0.10,
        help="Minimum gap between top-1 and top-2 class probabilities in sequence mode.",
    )
    parser.add_argument(
        "--min-event-windows",
        type=int,
        default=2,
        help="Minimum number of nearby windows required to keep an event.",
    )
    parser.add_argument(
        "--merge-gap",
        type=float,
        default=0.6,
        help="Merge same-label detections separated by less than this many seconds.",
    )
    parser.add_argument(
        "--include-hold",
        action="store_true",
        help="Include hold predictions in the printed sequence.",
    )
    parser.add_argument(
        "--include-stop",
        action="store_true",
        help="Include stop predictions in the printed sequence.",
    )
    parser.add_argument(
        "--max-commands",
        type=int,
        default=0,
        help="Limit output to the first N distinct commands. Use 2 for exactly two-command tests.",
    )
    args = parser.parse_args()

    if args.window_seconds <= 0:
        raise ValueError("--window-seconds must be > 0")
    if args.step_seconds <= 0:
        raise ValueError("--step-seconds must be > 0")
    if args.min_event_windows <= 0:
        raise ValueError("--min-event-windows must be > 0")
    if args.max_commands < 0:
        raise ValueError("--max-commands must be >= 0")
    if args.sequence_duration < args.window_seconds:
        raise ValueError("--sequence-duration must be >= --window-seconds")

    model = tf.keras.models.load_model(args.model)
    if args.mode == "single":
        predict_from_microphone(model, analysis_window_seconds=args.window_seconds)
    else:
        predict_sequence_from_microphone(
            model=model,
            total_duration=args.sequence_duration,
            analysis_window_seconds=args.window_seconds,
            step_seconds=args.step_seconds,
            min_confidence=args.min_confidence,
            min_margin=args.min_margin,
            min_event_windows=args.min_event_windows,
            merge_gap_seconds=args.merge_gap,
            include_hold=args.include_hold,
            include_stop=args.include_stop,
            max_commands=args.max_commands,
        )


if __name__ == "__main__":
    main()
