"""One-off comparative trial: before/after validation for the marker-target
jump-gate added to `host_software/src/state_machine.py`'s `_MarkerPositionFilter`
(2026-09-15, "Marker Target Jump-Gate" -- see docs/PROJECT_LOGBOOK.md).

Bug being validated: `TargetStateMachine.get_target_coords()` used to average a
plain `deque(maxlen=10)` of whatever `{color: (x, y)}` the marker classifier
reported each frame, with no outlier/jump rejection -- unlike `PredictionGate`,
which already gates the ball's own position this way. A sustained ~0.5-1s
marker-classifier misclassification burst (e.g. yellow_square's blob briefly
flickering to "green" -- see this session's earlier marker_classifier.py
entries) rides straight through that average and can move the commanded
target tens of mm, not just a scoring artifact.

Data available and its limitation (stated plainly, not glossed over): the three
real Jetson recordings analyzed here
(`host_software/data/01_bronze/evaluation/ground_truth_jetson_20260915_*.csv`)
log `target_x_mm`/`target_y_mm` -- i.e. `get_target_coords()`'s OLD, ungated
OUTPUT -- every frame. They do NOT log the raw per-color marker detections that
fed that average (no per-color HSV/blob log exists for these runs). So a
byte-for-byte replay of the new gate (which runs on raw per-color candidates,
before averaging) isn't possible against this data. What IS possible, and what
this script does: identify each recording's real marker-following phases
(go_green/go_yellow/go_red/go_black dwells, matched against the sheet's known
touch-mm marker positions -- see MARKER_POSITIONS_MM below) and, within each
one, treat the recorded (already-once-averaged) target trajectory as the best
available proxy input, then run it through a *fresh* `_MarkerPositionFilter`
per phase and compare frame-to-frame step-size statistics before vs. after.
This is a conservative proxy in one specific sense: a raw contamination sample
gated out BEFORE ever entering the old average would never even appear in this
already-averaged proxy signal at full strength (see module docstring in
state_machine.py for why a full-window burst can still swing the mean by its
full raw distance) -- so what we can observe here is what the gate does to a
signal that has already partly absorbed the contamination, which is the
harder case, not the easy one.

Run as a module from the repo root:

    python -m host_software.ml_vision.experiments.marker_target_jump_gate_eval
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from host_software.evaluations.evaluate_system_control import (
    load_telemetry,
    segment_by_target,
    TARGET_CHANGE_TOLERANCE_MM,
    MIN_TRIAL_DURATION_MS,
)
from host_software.src.state_machine import _MarkerPositionFilter

REPO_ROOT = Path(__file__).resolve().parents[3]
CSV_PATHS = [
    REPO_ROOT / "host_software/data/01_bronze/evaluation/ground_truth_jetson_20260915_150041.csv",
    REPO_ROOT / "host_software/data/01_bronze/evaluation/ground_truth_jetson_20260915_150729.csv",
    REPO_ROOT / "host_software/data/01_bronze/evaluation/ground_truth_jetson_20260915_151627.csv",
]

# touch-mm positions of the aruco_markers_03 sheet's 4 evaluation-sequence
# colors (docs/EVALUATION_STRATEGY.md, updated 2026-09-15: green/red/yellow/
# black only). Derived from aruco_markers_03_manifest.json's manifest_mm
# center_mm via the project's standard touch_x = W/2 - manifest_x,
# touch_y = manifest_y - H/2 convention (marker_classifier.py /
# main_onnx_shared_vision_audio.py's px_to_touch_mm docstring):
#   green_hexagon  manifest (93.75, 101.0) -> touch (0.0,  30.0)
#   yellow_square  manifest (123.75, 71.0) -> touch (-30.0, 0.0)
#   red_triangle   manifest (63.75, 71.0)  -> touch (30.0,  0.0)
#   black_circle   manifest (93.75, 71.0)  -> touch (0.0,   0.0)  <- == center!
MARKER_POSITIONS_MM: Dict[str, Tuple[float, float]] = {
    "green": (0.0, 30.0),
    "yellow": (-30.0, 0.0),
    "red": (30.0, 0.0),
    "black": (0.0, 0.0),
}
# Half the sheet's 30mm feature spacing -- same match-radius convention used by
# marker_classifier_green_fallback_sweep.py's nearest-feature matching.
MATCH_RADIUS_MM = 15.0


def _label_segments(df: pd.DataFrame, segments: List[Tuple[int, int]]) -> List[Optional[str]]:
    """Match each merged (real-dwell) segment's mean target position against
    MARKER_POSITIONS_MM. "black" and "center" are both (0, 0) on this sheet
    (black_circle sits at the platform's exact physical center -- a real,
    already-logged coincidence, not a bug in this script) -- disambiguated by
    order: a (0,0) segment is only labeled "black" if it starts after at least
    one green/yellow/red segment has already ended, since a real go_black
    command necessarily follows those in docs/EVALUATION_STRATEGY.md's
    Standardized Evaluation Sequence. An earlier (0,0) segment is the
    pre-sequence center/hold phase, not a real go_black -- labeled None
    (excluded from the marker-following analysis) rather than guessed at."""
    labels: List[Optional[str]] = [None] * len(segments)
    seen_marker_color = False
    for i, (start, end) in enumerate(segments):
        mean_x = float(df["target_x"].iloc[start:end].mean())
        mean_y = float(df["target_y"].iloc[start:end].mean())
        best_color, best_dist = None, MATCH_RADIUS_MM
        for color, (mx, my) in MARKER_POSITIONS_MM.items():
            dist = math.hypot(mean_x - mx, mean_y - my)
            if dist <= best_dist:
                best_color, best_dist = color, dist
        if best_color in ("green", "yellow", "red"):
            labels[i] = best_color
            seen_marker_color = True
        elif best_color == "black":
            labels[i] = "black" if seen_marker_color else None
    return labels


def _step_stats(xs: np.ndarray, ys: np.ndarray) -> Dict[str, float]:
    if len(xs) < 2:
        return {"max_mm": 0.0, "p95_mm": 0.0, "mean_mm": 0.0, "n_over_5mm": 0, "argmax": -1}
    dx = np.diff(xs)
    dy = np.diff(ys)
    steps = np.hypot(dx, dy)
    return {
        "max_mm": float(steps.max()),
        "p95_mm": float(np.percentile(steps, 95)),
        "mean_mm": float(steps.mean()),
        "n_over_5mm": int(np.sum(steps > 5.0)),
        "argmax": int(np.argmax(steps)) + 1,  # +1: index into xs/ys of the landing frame
    }


def _replay_through_gate(xs: np.ndarray, ys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Feed a recorded (already-once-averaged) target trajectory through a
    fresh _MarkerPositionFilter, frame by frame, as the closest available
    proxy for what the new per-color gate does to this signal (see module
    docstring for why an exact raw-sample replay isn't possible from this
    data). Frames before the filter seeds (first seed_window consistent
    samples) hold at the first raw sample rather than an undefined value, so
    the two series stay directly comparable frame-for-frame."""
    # history_size=10 matches TargetStateMachine's own default constructor
    # exactly (it always passes history_size explicitly, overriding this
    # class's own default of 30) -- faithful to shipped production behavior,
    # not a more favorable number picked for this harness.
    filt = _MarkerPositionFilter(history_size=10)
    out_x = np.empty(len(xs))
    out_y = np.empty(len(ys))
    for i in range(len(xs)):
        filt.update(xs[i], ys[i])
        pos = filt.position
        if pos is None:
            out_x[i], out_y[i] = xs[i], ys[i]
        else:
            out_x[i], out_y[i] = pos
    return out_x, out_y


def analyze_csv(csv_path: Path) -> None:
    print(f"\n{'=' * 88}\n{csv_path.name}\n{'=' * 88}")
    df = load_telemetry(str(csv_path))

    merged = segment_by_target(df, tol_mm=TARGET_CHANGE_TOLERANCE_MM, min_duration_ms=MIN_TRIAL_DURATION_MS)
    labels = _label_segments(df, merged)

    n_marker_phases = sum(1 for lbl in labels if lbl is not None)
    if n_marker_phases == 0:
        print("  No green/yellow/red/black marker-following phase matched in this recording.")
        return

    for (start, end), label in zip(merged, labels):
        if label is None:
            continue
        seg = df.iloc[start:end]
        raw_x = seg["target_x"].to_numpy(dtype=float)
        raw_y = seg["target_y"].to_numpy(dtype=float)
        t0, t1 = seg["timestamp_ms"].iloc[0], seg["timestamp_ms"].iloc[-1]

        before = _step_stats(raw_x, raw_y)
        gated_x, gated_y = _replay_through_gate(raw_x, raw_y)
        after = _step_stats(gated_x, gated_y)

        print(
            f"\n  go_{label:<6s} phase: {len(seg)} frames, {t1 - t0:.0f}ms "
            f"(rows {start}:{end})"
        )
        print(
            f"    BEFORE (raw target_x/y, current shipped behavior): "
            f"max={before['max_mm']:.1f}mm  p95={before['p95_mm']:.1f}mm  "
            f"mean={before['mean_mm']:.2f}mm  frames>5mm={before['n_over_5mm']}"
        )
        print(
            f"    AFTER  (replayed through _MarkerPositionFilter):   "
            f"max={after['max_mm']:.1f}mm  p95={after['p95_mm']:.1f}mm  "
            f"mean={after['mean_mm']:.2f}mm  frames>5mm={after['n_over_5mm']}"
        )
        if before["max_mm"] > 0:
            reduction_pct = 100.0 * (1.0 - after["max_mm"] / before["max_mm"])
            note = ""
            if reduction_pct < 1.0 and before["argmax"] <= 5:
                note = (
                    "  (spike lands within the first seed_window=5 frames of this "
                    "color's session -- pre-confirmation frames pass through "
                    "unfiltered by design, same as PredictionGate's own seed phase; "
                    "see report)"
                )
            print(f"    max-step reduction: {reduction_pct:.1f}%{note}")


def main() -> None:
    for csv_path in CSV_PATHS:
        if not csv_path.exists():
            print(f"skip (not found): {csv_path}")
            continue
        analyze_csv(csv_path)


if __name__ == "__main__":
    main()
