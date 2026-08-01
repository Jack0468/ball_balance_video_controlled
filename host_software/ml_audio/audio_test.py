#!/usr/bin/env python3
"""
Standalone microphone test for the 6-word audio command classifier.

Captures audio continuously and prints one command per window (default 2 s).
Prints "hold" when the window is silent, has no detectable speech, or the
model's confidence falls below the threshold.

Self-contained: the network is defined here and the weights are loaded from
your .pth state_dict, so this file does not import anything from the project.

Usage:
    python mic_command_test.py --weights path/to/audio_command_classifier_state_dict.pth
    python mic_command_test.py --weights ... --verbose
    python mic_command_test.py --list-devices
"""

import argparse
import json
import queue
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import sounddevice as sd

# ----------------------------------------------------------------------------
# Constants -- these must match the training notebook exactly.
# ----------------------------------------------------------------------------

SAMPLE_RATE = 16_000
CLIP_SECONDS = 1.25  # what the model was trained on
TARGET_SAMPLES = int(SAMPLE_RATE * CLIP_SECONDS)  # 20000

# Training used tf.signal.stft(frame_length=255, frame_step=128).
# TF rounds fft_length up to the next power of two, so it is really a
# 256-point FFT over a 255-sample Hann window -> 129 frequency bins.
N_FFT = 256
WIN_LENGTH = 255
HOP_LENGTH = 128

LABELS = ["go_blue", "go_green", "go_red", "go_yellow", "hold", "stop"]
LABELS_WITH_BACKGROUND = [
    "_background_",
    "go_blue",
    "go_green",
    "go_red",
    "go_yellow",
    "hold",
    "stop",
]
LABELS_12 = [
    "_background_",
    "go_blue",
    "go_green",
    "go_red",
    "go_yellow",
    "hold",
    "stop",
    "go_grey",
    "forward",
    "backward",
    "left",
    "right",
]

# Keras defaults; these are plain attributes in the generated module, so they
# are NOT part of the state_dict and have to be restated here.
BN_EPS = 1e-3
NORM_EPS = 1e-7


# ----------------------------------------------------------------------------
# Model -- mirrors audio_command_classifier_pytorch.py exactly.
# ----------------------------------------------------------------------------


class AudioCommandClassifier(torch.nn.Module):
    """Conv12 -> Conv24 -> Conv48 -> global average pool -> Dense(6)."""

    def __init__(self, num_classes=6):
        super().__init__()
        self.num_classes = int(num_classes)
        self.norm_epsilon = NORM_EPS
        self.bn1_eps = BN_EPS
        self.bn2_eps = BN_EPS
        self.bn3_eps = BN_EPS

        self.register_buffer("norm_mean", torch.zeros(1))
        self.register_buffer("norm_variance", torch.ones(1))

        self.register_buffer("conv1_weight", torch.zeros(12, 1, 3, 3))
        self.register_buffer("conv1_bias", torch.zeros(12))
        self.register_buffer("bn1_gamma", torch.ones(12))
        self.register_buffer("bn1_beta", torch.zeros(12))
        self.register_buffer("bn1_mean", torch.zeros(12))
        self.register_buffer("bn1_var", torch.ones(12))

        self.register_buffer("conv2_weight", torch.zeros(24, 12, 3, 3))
        self.register_buffer("conv2_bias", torch.zeros(24))
        self.register_buffer("bn2_gamma", torch.ones(24))
        self.register_buffer("bn2_beta", torch.zeros(24))
        self.register_buffer("bn2_mean", torch.zeros(24))
        self.register_buffer("bn2_var", torch.ones(24))

        self.register_buffer("conv3_weight", torch.zeros(48, 24, 3, 3))
        self.register_buffer("conv3_bias", torch.zeros(48))
        self.register_buffer("bn3_gamma", torch.ones(48))
        self.register_buffer("bn3_beta", torch.zeros(48))
        self.register_buffer("bn3_mean", torch.zeros(48))
        self.register_buffer("bn3_var", torch.ones(48))

        self.register_buffer("dense_weight", torch.zeros(self.num_classes, 48))
        self.register_buffer("dense_bias", torch.zeros(self.num_classes))

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)

        if x.shape[-2:] != (64, 64):
            x = F.interpolate(x, size=(64, 64), mode="bilinear", align_corners=False)

        x = (x - self.norm_mean.view(1, -1, 1, 1)) / torch.sqrt(
            self.norm_variance.view(1, -1, 1, 1) + self.norm_epsilon
        )

        x = F.conv2d(x, self.conv1_weight, bias=self.conv1_bias, padding=1)
        x = F.relu(x)
        x = F.batch_norm(
            x,
            self.bn1_mean,
            self.bn1_var,
            self.bn1_gamma,
            self.bn1_beta,
            training=False,
            eps=self.bn1_eps,
        )
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.conv2d(x, self.conv2_weight, bias=self.conv2_bias, padding=1)
        x = F.relu(x)
        x = F.batch_norm(
            x,
            self.bn2_mean,
            self.bn2_var,
            self.bn2_gamma,
            self.bn2_beta,
            training=False,
            eps=self.bn2_eps,
        )
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.conv2d(x, self.conv3_weight, bias=self.conv3_bias, padding=1)
        x = F.relu(x)
        x = F.batch_norm(
            x,
            self.bn3_mean,
            self.bn3_var,
            self.bn3_gamma,
            self.bn3_beta,
            training=False,
            eps=self.bn3_eps,
        )

        x = x.mean(dim=(2, 3))
        return F.linear(x, self.dense_weight, self.dense_bias)


def _labels_for_num_classes(num_classes: int):
    if num_classes == 6:
        return LABELS
    if num_classes == 7:
        return LABELS_WITH_BACKGROUND
    if num_classes == 12:
        return LABELS_12
    return [f"class_{i}" for i in range(num_classes)]


def load_model(weights_path):
    """Load a state_dict .pth, or fall back to a TorchScript .pt."""
    try:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
    except Exception:
        print("Not a plain state_dict, trying TorchScript...", file=sys.stderr)
        scripted = torch.jit.load(weights_path, map_location="cpu")
        scripted.eval()
        return scripted, LABELS

    if not isinstance(state, dict):
        raise SystemExit(f"Expected a state_dict in {weights_path}, got {type(state)}")

    state = {k.replace("module.", "", 1): v for k, v in state.items()}

    num_classes = int(state["dense_bias"].shape[0]) if "dense_bias" in state else 6
    model = AudioCommandClassifier(num_classes=num_classes)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print("State_dict did not match the expected layout.", file=sys.stderr)
        if missing:
            print(f"  missing:    {sorted(missing)}", file=sys.stderr)
        if unexpected:
            print(f"  unexpected: {sorted(unexpected)}", file=sys.stderr)
        print(f"  file keys:  {sorted(state.keys())}", file=sys.stderr)
        raise SystemExit("Aborting: weights would be partially uninitialised.")

    labels_path = Path(weights_path).with_name("labels.json")
    labels = None
    if labels_path.exists():
        try:
            loaded = json.loads(labels_path.read_text(encoding="utf-8"))
            if isinstance(loaded, list) and len(loaded) == num_classes:
                labels = loaded
        except Exception:
            labels = None

    if labels is None:
        labels = _labels_for_num_classes(num_classes)

    model.eval()
    return model, labels


# ----------------------------------------------------------------------------
# Preprocessing -- same as the bronze -> silver step that built the train set.
# ----------------------------------------------------------------------------


def align_speech_to_fixed_length(audio, target_samples=TARGET_SAMPLES):
    """Trim to the speech region, pad/crop to 1.25 s, peak-normalise.

    Returns (waveform, reason). waveform is None when nothing usable is found.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    rms = float(np.sqrt(np.mean(audio**2))) if audio.size else 0.0

    if peak < 0.03 or rms < 0.003:
        return None, "too_quiet"

    threshold = max(0.015, peak * 0.08)
    active = np.where(np.abs(audio) > threshold)[0]
    if len(active) == 0:
        return None, "no_speech"

    start = max(0, active[0] - int(0.08 * SAMPLE_RATE))
    end = min(len(audio), active[-1] + int(0.12 * SAMPLE_RATE))
    audio = audio[start:end]

    if len(audio) > target_samples:
        audio = audio[:target_samples]
    if len(audio) < target_samples:
        audio = np.pad(audio, (0, target_samples - len(audio)))

    peak = float(np.max(np.abs(audio)))
    if peak > 1e-6:
        audio = audio / peak * 0.95

    return audio.astype(np.float32), "ok"


_WINDOW = torch.hann_window(WIN_LENGTH, periodic=True)


def waveform_to_spectrogram(waveform):
    """Log-magnitude STFT matching tf.signal.stft(255, 128). -> [1, 1, 155, 129]"""
    x = torch.from_numpy(np.ascontiguousarray(waveform))
    spec = torch.stft(
        x,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=_WINDOW,
        center=False,
        return_complex=True,
    )
    spec = torch.log(spec.abs() + 1e-6)
    return spec.transpose(0, 1).unsqueeze(0).unsqueeze(0)


def predict_window(model, labels, window, threshold, verbose):
    stamp = time.strftime("%H:%M:%S")
    aligned, reason = align_speech_to_fixed_length(window)

    if aligned is None:
        print(f"[{stamp}] hold          ({reason})")
        return

    spec = waveform_to_spectrogram(aligned)
    with torch.no_grad():
        probs = torch.softmax(model(spec), dim=-1)[0].numpy()

    top = int(np.argmax(probs))
    label = labels[top]
    conf = float(probs[top])

    if conf < threshold:
        print(f"[{stamp}] hold          " f"(top={label} {conf:.2f} below threshold)")
    else:
        print(f"[{stamp}] {label:<13} conf={conf:.2f}")

    if verbose:
        ranked = sorted(zip(labels, probs), key=lambda p: -p[1])
        print("            " + "  ".join(f"{n}={p:.3f}" for n, p in ranked))


# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------


def run(args):
    model, labels = load_model(args.weights)
    print(f"Loaded weights from {args.weights}")

    device_info = sd.query_devices(
        args.device if args.device is not None else sd.default.device[0], "input"
    )
    print(f"Microphone: {device_info['name']}")
    print(f"Window: {args.window:.1f} s   confidence threshold: {args.threshold:.2f}")
    if args.single_shot:
        print("Single-shot mode: mic records only during each capture window.")
        print("Press Enter to record once, or type q then Enter to quit.\n")
    else:
        print("Listening continuously. Ctrl+C to stop.\n")

    window_samples = int(SAMPLE_RATE * args.window)

    if args.single_shot:
        try:
            while True:
                cmd = input("Record next command? [Enter/q]: ").strip().lower()
                if cmd in {"q", "quit", "exit"}:
                    print("Stopped.")
                    return

                recording = sd.rec(
                    window_samples,
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    dtype="float32",
                    device=args.device,
                )
                sd.wait()
                window = np.squeeze(recording, axis=-1)
                predict_window(model, labels, window, args.threshold, args.verbose)
        except KeyboardInterrupt:
            print("\nStopped.")
        return

    chunks = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"  [stream: {status}]", file=sys.stderr)
        chunks.put(indata.copy().reshape(-1))

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=args.device,
        blocksize=int(SAMPLE_RATE * 0.1),
        callback=callback,
    )

    pending = np.zeros(0, dtype=np.float32)

    with stream:
        try:
            while True:
                # Accumulate a full window without dropping audio between them.
                while len(pending) < window_samples:
                    pending = np.concatenate([pending, chunks.get()])

                window = pending[:window_samples]
                pending = pending[window_samples:]

                predict_window(model, labels, window, args.threshold, args.verbose)

        except KeyboardInterrupt:
            print("\nStopped.")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--weights", help="Path to the .pth state_dict.")
    parser.add_argument(
        "--window",
        type=float,
        default=2.0,
        help="Seconds of audio per decision (default: 2.0).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.60,
        help="Below this confidence, print hold (default: 0.60).",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Input device index (see --list-devices).",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Also print the full probability vector."
    )
    parser.add_argument(
        "--single-shot",
        action="store_true",
        help="Record only one window each time you press Enter (mic is not continuously on).",
    )
    parser.add_argument(
        "--list-devices", action="store_true", help="List audio devices and exit."
    )
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    if not args.weights:
        parser.error("--weights is required (or use --list-devices)")

    run(args)


if __name__ == "__main__":
    main()
