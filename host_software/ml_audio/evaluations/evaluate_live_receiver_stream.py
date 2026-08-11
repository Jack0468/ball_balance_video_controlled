"""Validate the live AudioCommandReceiver end-to-end against a continuous stream.

Unlike evaluate_audio_classifier.py (per-clip, offline), this drives the real
receiver code (rolling buffer, confidence/margin gating, everything) through
data/02_silver/master_evaluation_audio.wav via its source_file playback mode,
so it's testing the actual live decision logic, not a reimplementation of it.

master_evaluation_audio.wav was built by create_master_audio.py, which
overlays one command every 10s onto looped background noise, in this known
order (see that script's EVAL_SEQUENCE):
    0s go_grey, 10s go_blue, 20s go_green, 30s go_yellow, 40s go_red,
    50s forward, 60s left, 70s right, 80s backward, 90s hold, 100s stop,
    110s background (nothing overlaid)

This script polls get_latest_command() for the duration of the file and
reports which of those 11 spoken commands were correctly detected within
their window, in order -- the metric that actually matters for the "produces
incorrect outputs during concurrent operation" bug this whole investigation
started from. Takes ~2 minutes wall-clock since the receiver simulates
real-time cadence on purpose (that's the point of testing it this way).
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ML_AUDIO_DIR = os.path.dirname(SCRIPT_DIR)
HOST_SOFTWARE_DIR = os.path.dirname(ML_AUDIO_DIR)
if HOST_SOFTWARE_DIR not in sys.path:
    sys.path.insert(0, HOST_SOFTWARE_DIR)

from ml_audio.audio_receiver_pytorch import AudioCommandReceiver  # noqa: E402

DEFAULT_MODEL = os.path.join(
    ML_AUDIO_DIR, "models", "pytorch_v3", "audio_command_classifier_state_dict_v3.pth"
)
DEFAULT_STREAM = os.path.join(ML_AUDIO_DIR, "data", "02_silver", "master_evaluation_audio.wav")
DEFAULT_REPORT_DIR = os.path.join(SCRIPT_DIR, "reports")

EXPECTED_SEQUENCE = [
    (0, "go_grey"),
    (10, "go_blue"),
    (20, "go_green"),
    (30, "go_yellow"),
    (40, "go_red"),
    (50, "forward"),
    (60, "left"),
    (70, "right"),
    (80, "backward"),
    (90, "hold"),
    (100, "stop"),
]
STREAM_DURATION_SEC = 120


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Drive the live AudioCommandReceiver through a continuous stream."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--stream", default=DEFAULT_STREAM)
    args = parser.parse_args()

    receiver = AudioCommandReceiver(args.model, source_file=args.stream)

    # (elapsed_seconds, command) for every distinct command the receiver latched.
    detections: list[tuple[float, str]] = []
    start = time.perf_counter()
    while time.perf_counter() - start < STREAM_DURATION_SEC + 3:
        cmd = receiver.get_latest_command()
        if cmd is not None:
            detections.append((time.perf_counter() - start, cmd))
        time.sleep(0.05)
    receiver.stop()

    # For each expected command, did the receiver detect it at some point
    # within its 10s window (plus a little slack for latency)?
    results = []
    for window_start, expected in EXPECTED_SEQUENCE:
        window_end = window_start + 10 + 2  # 2s slack for detection latency
        hits = [
            (t, cmd)
            for t, cmd in detections
            if window_start <= t < window_end and cmd == expected
        ]
        results.append(
            {
                "window_start_sec": window_start,
                "expected": expected,
                "detected": hits[0][1] if hits else None,
                "detected_at_sec": round(hits[0][0], 2) if hits else None,
                "correct": bool(hits),
            }
        )

    correct = sum(1 for r in results if r["correct"])
    total = len(results)

    os.makedirs(DEFAULT_REPORT_DIR, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = os.path.join(DEFAULT_REPORT_DIR, f"live_stream_eval_{timestamp}.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "generated_at": timestamp,
                "model": os.path.relpath(args.model, ML_AUDIO_DIR),
                "stream": os.path.relpath(args.stream, ML_AUDIO_DIR),
                "correct": correct,
                "total": total,
                "results": results,
                "all_detections": [{"t": round(t, 2), "cmd": c} for t, c in detections],
            },
            f,
            indent=2,
        )

    print(f"\n{correct}/{total} expected commands correctly detected in their window\n")
    for r in results:
        status = "OK  " if r["correct"] else "MISS"
        detected = r["detected"] or "-"
        print(
            f"  [{status}] t={r['window_start_sec']:>3d}s expected={r['expected']:<10s} "
            f"detected={detected}"
        )
    print(f"\nFull report: {out_path}")


if __name__ == "__main__":
    main()
