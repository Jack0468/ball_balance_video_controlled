"""Diagnostic: run a NeMo checkpoint over master_evaluation_audio.wav at the
exact same window/step cadence as NemoAudioCommandReceiver, but log every
window's full prediction (top label, confidence, margin) rather than only
the ones that pass the confidence/margin gate.

Built to answer a specific question from the live-stream result: for a
window that missed (no detection, or a wrong detection), was the model
actually close -- right label but just under threshold -- or genuinely
wrong? That distinction decides whether a fix is a one-line gate-threshold
tweak or an actual model/data problem. Not run in a live thread (no
real-time sleep needed for offline analysis), so this finishes in seconds
rather than the ~2 minutes the real receiver takes.

Usage:
    python nemo_stream_probe.py
    python nemo_stream_probe.py --model <path> --window-start 80 --window-end 90
"""

import argparse
import os
import sys

import numpy as np
import soundfile as sf
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ML_AUDIO_DIR = os.path.dirname(SCRIPT_DIR)
HOST_SOFTWARE_DIR = os.path.dirname(ML_AUDIO_DIR)
if HOST_SOFTWARE_DIR not in sys.path:
    sys.path.insert(0, HOST_SOFTWARE_DIR)

from ml_audio.audio_dsp import OUTPUT_SEQUENCE_LENGTH, SAMPLE_RATE  # noqa: E402

DEFAULT_MODEL = os.path.join(ML_AUDIO_DIR, "models", "nemo_matchboxnet_v1", "matchboxnet_finetuned.nemo")
DEFAULT_STREAM = os.path.join(ML_AUDIO_DIR, "data", "02_silver", "master_evaluation_audio.wav")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Log every window's raw prediction/confidence/margin over the eval stream."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--stream", default=DEFAULT_STREAM)
    parser.add_argument("--step-seconds", type=float, default=0.2)
    parser.add_argument("--min-confidence", type=float, default=0.8)
    parser.add_argument("--min-margin", type=float, default=0.15)
    parser.add_argument("--window-start", type=float, default=0.0, help="only print windows at/after this stream time (sec)")
    parser.add_argument("--window-end", type=float, default=120.0, help="only print windows before this stream time (sec)")
    args = parser.parse_args()

    import nemo.collections.asr as nemo_asr

    print(f"Loading {args.model}...")
    model = nemo_asr.models.EncDecClassificationModel.restore_from(args.model)
    model.eval()
    labels = list(model.cfg.labels)  # NOT model.labels -- see nemo_live_receiver.py's docstring
    print(f"Labels ({len(labels)}): {labels}")

    audio, sr = sf.read(args.stream, dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != SAMPLE_RATE:
        from scipy.signal import resample
        audio = resample(audio, int(len(audio) * SAMPLE_RATE / sr)).astype(np.float32)

    window_samples = OUTPUT_SEQUENCE_LENGTH
    step_samples = int(SAMPLE_RATE * args.step_seconds)
    buffer = np.zeros(window_samples, dtype=np.float32)

    last_pushed = None
    idx = 0
    print(f"\n{'t':>7s}  {'top_label':<14s} {'conf':>6s} {'margin':>7s}  {'gate':>6s}  {'pushed':<14s}")
    with torch.no_grad():
        while idx < len(audio):
            end_idx = min(idx + step_samples, len(audio))
            chunk = audio[idx:end_idx]
            if len(chunk) < step_samples:
                chunk = np.pad(chunk, (0, step_samples - len(chunk)))

            buffer = np.roll(buffer, -len(chunk))
            buffer[-len(chunk):] = chunk

            t = idx / SAMPLE_RATE
            idx += step_samples

            if not (args.window_start <= t < args.window_end):
                continue

            audio_t = torch.as_tensor(buffer, dtype=torch.float32).unsqueeze(0)
            len_t = torch.tensor([window_samples])
            logits = model(input_signal=audio_t, input_signal_length=len_t)
            probs = torch.nn.functional.softmax(logits, dim=-1)[0].numpy()

            top_id = int(np.argmax(probs))
            top_label = labels[top_id]
            top_conf = float(probs[top_id])
            top_two = np.partition(probs, -2)[-2:]
            margin = float(top_two[-1] - top_two[-2])

            gate_pass = top_conf >= args.min_confidence and margin >= args.min_margin
            pushed = ""
            if gate_pass and top_label != last_pushed:
                pushed = top_label
                last_pushed = top_label
            elif not gate_pass:
                last_pushed = None

            gate_str = "PASS" if gate_pass else "-"
            print(f"{t:7.2f}  {top_label:<14s} {top_conf:6.3f} {margin:7.3f}  {gate_str:>6s}  {pushed:<14s}")


if __name__ == "__main__":
    main()
