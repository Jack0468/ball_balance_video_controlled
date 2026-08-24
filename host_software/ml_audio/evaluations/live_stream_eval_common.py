"""Receiver-agnostic live-stream evaluation: poll a receiver through
master_evaluation_audio.wav and score which expected commands landed in
their window.

Factored out of evaluate_live_receiver_stream.py so the pretrained-backbone
track (NeMo, see docs/plans/audio_eval_notebook_refactor_plan.md,
"Pretrained Backbone / Transfer Learning Track") can reuse the exact same
scoring methodology via evaluate_nemo_live_receiver_stream.py, rather than
maintaining a second copy of the window-matching logic that could quietly
drift from this one. Deliberately does NOT import audio_receiver_pytorch
(needs sounddevice) or nemo -- this module only needs `receiver` to expose
get_latest_command()/stop(), so it stays importable in both the local
(sounddevice-only) and Colab (NeMo-only) environments.

master_evaluation_audio.wav was built by create_master_audio.py, which
overlays one command every 10s onto looped background noise, in this known
order (see that script's EVAL_SEQUENCE):
    0s go_grey, 10s go_blue, 20s go_green, 30s go_yellow, 40s go_red,
    50s forward, 60s left, 70s right, 80s backward, 90s hold, 100s stop,
    110s background (nothing overlaid)

`go_grey` (0s) is deliberately excluded from EXPECTED_SEQUENCE below -- the
current robot deployment has no grey marker to test against, so it isn't
part of the scored set for now (still spoken in the stream audio itself,
just not graded). Live-stream scores from before this change were out of
11; from this change onward they're out of 10. Not a big swing in practice
-- go_grey scored correctly in all 3 NeMo seeds tested so far, so dropping
it doesn't change which classes look weak (go_green/backward still the
consistent failures).
"""

import json
import os
import time
from datetime import datetime, timezone

EXPECTED_SEQUENCE = [
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


def run_live_stream_eval(
    receiver, model_label: str, stream_label: str, out_dir: str
) -> dict:
    """Polls `receiver.get_latest_command()` for the duration of the stream,
    scores against EXPECTED_SEQUENCE, writes a report to out_dir in the same
    format every checkpoint in this plan (v3-v7) has been evaluated with,
    and returns that report dict. `receiver` just needs get_latest_command()
    and stop() -- works with AudioCommandReceiver, NemoAudioCommandReceiver,
    or anything else with that shape."""
    detections: list[tuple[float, str]] = []
    start = time.perf_counter()
    while time.perf_counter() - start < STREAM_DURATION_SEC + 3:
        cmd = receiver.get_latest_command()
        if cmd is not None:
            detections.append((time.perf_counter() - start, cmd))
        time.sleep(0.05)
    receiver.stop()

    results = []
    for window_start, expected in EXPECTED_SEQUENCE:
        window_end = window_start + 10 + 2  # 2s slack for detection latency
        hits = [
            (t, cmd)
            for t, cmd in detections
            if window_start <= t < window_end and cmd == expected
        ]
        results.append({
            "window_start_sec": window_start,
            "expected": expected,
            "detected": hits[0][1] if hits else None,
            "detected_at_sec": round(hits[0][0], 2) if hits else None,
            "correct": bool(hits),
        })

    correct = sum(1 for r in results if r["correct"])
    total = len(results)

    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = os.path.join(out_dir, f"live_stream_eval_{timestamp}.json")
    report = {
        "generated_at": timestamp,
        "model": model_label,
        "stream": stream_label,
        "correct": correct,
        "total": total,
        "results": results,
        "all_detections": [{"t": round(t, 2), "cmd": c} for t, c in detections],
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    report["_report_path"] = out_path

    print(f"\n{correct}/{total} expected commands correctly detected in their window\n")
    for r in results:
        status = "OK  " if r["correct"] else "MISS"
        detected = r["detected"] or "-"
        print(
            f"  [{status}] t={r['window_start_sec']:>3d}s expected={r['expected']:<10s} "
            f"detected={detected}"
        )
    print(f"\nFull report: {out_path}")

    return report
