"""Find and verify the USB camera's built-in microphone, on either the laptop
(Windows) or the Jetson (Linux) -- same script, cross-platform via sounddevice.

Confirmed 2026-09-15 (laptop): AudioCommandReceiverONNX's sd.InputStream() call
takes no `device` argument, so it silently relies on whatever the OS considers
the "default" input device. On this laptop that currently happens to be the
camera's mic (JBCW036) -- but that's an OS setting, not something this codebase
controls, and Linux/the Jetson will enumerate devices completely differently.
This script finds the camera mic explicitly by name match instead of trusting
whatever the system default happens to be.

Usage:
    python test_microphone.py [--name JBCW036] [--device INDEX] [--seconds 3]
"""

import argparse
import sys
from typing import Optional

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16_000

# Same thresholds AudioCommandReceiverONNX.align_speech_to_fixed_length uses to
# decide "silence, discard" vs. "real speech, process" -- reuse them here so a
# passing result here actually means the real pipeline would accept this audio.
PEAK_THRESHOLD = 0.03
RMS_THRESHOLD = 0.003


def find_device_by_name(name_substring: str) -> Optional[int]:
    devices = sd.query_devices()
    matches = [
        i
        for i, d in enumerate(devices)
        if name_substring.lower() in d["name"].lower() and d["max_input_channels"] > 0
    ]
    if not matches:
        return None
    # Prefer WASAPI/DirectSound-style entries over MME when several APIs expose
    # the same physical device -- MME is listed first by sounddevice but is the
    # oldest/most limited Windows audio API; doesn't matter on Linux (ALSA only
    # shows one entry per device there).
    return matches[-1]


def record_and_check(device_index: int, seconds: float) -> None:
    info = sd.query_devices(device_index)
    print(f"Recording {seconds}s from device {device_index}: {info['name']}")
    print("Speak a command now (e.g. 'go red')...")

    recording = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=device_index,
    )
    sd.wait()

    audio = recording.flatten()
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio**2)))

    print(f"\nPeak amplitude: {peak:.4f} (need > {PEAK_THRESHOLD})")
    print(f"RMS amplitude:  {rms:.4f} (need > {RMS_THRESHOLD})")

    if peak < PEAK_THRESHOLD or rms < RMS_THRESHOLD:
        print(
            "\nFAIL -- align_speech_to_fixed_length() would treat this as silence "
            "and discard it. Check mic gain/OS input volume, or that you're "
            "actually speaking into the right physical mic."
        )
        sys.exit(1)
    else:
        print("\nPASS -- audio level is sufficient for the real pipeline to process it.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--name",
        default="JBCW036",
        help="Substring to match the camera mic's device name (default: JBCW036)",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Explicit device index, bypassing name search",
    )
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument(
        "--list", action="store_true", help="Just list devices and exit"
    )
    args = parser.parse_args()

    if args.list:
        print(sd.query_devices())
        return

    if args.device is not None:
        device_index = args.device
    else:
        device_index = find_device_by_name(args.name)
        if device_index is None:
            print(f"No input device matching '{args.name}' found. Devices:")
            print(sd.query_devices())
            sys.exit(1)

    record_and_check(device_index, args.seconds)


if __name__ == "__main__":
    main()
