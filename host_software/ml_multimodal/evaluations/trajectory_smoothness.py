"""Ball-trajectory jerk/smoothness -- the first of the two Extended Metrics flagged
2026-09-18 (`docs/EVALUATION_STRATEGY.md`'s "Extended Metrics" section;
`docs/PROJECT_LOGBOOK.md`'s 18/09/2026 "Telemetry Field-History Audit" entry) as "need
either new instrumentation or a dedicated test protocol, not just a new formula over
existing telemetry."

**That blanket statement does not hold for THIS metric, checked concretely (2026-10-13),
only for its sibling (disturbance-rejection/robustness, which genuinely does need a
dedicated hardware perturbation protocol and is explicitly out of scope here).** Jerk is
the 3rd time-derivative of ball position. Every telemetry schema
`host_software/evaluations/evaluate_system_control.py` already accepts carries exactly
the two raw signals a jerk computation needs and nothing else: a ball position
(`touch_x`/`touch_y`, already in its own `REQUIRED_COLUMNS`) and a per-frame timestamp
(`timestamp_ms`, already normalized by its own `load_telemetry()` from whichever
candidate column a given CSV uses). No new instrumentation, no higher sampling rate,
no firmware change -- a pure formula over what is logged today.

**Why this lives here, not in `evaluate_system_control.py` itself**: this agent's write
boundary is `host_software/ml_multimodal/` and `host_software/ml_jetson_vla/` only --
`host_software/evaluations/` is explicitly read-only ("for reference/integration",
`.claude/agents/ml-multimodal.md`). This module is written to be a drop-in addition:
`jerk_metrics_for_segments()` takes exactly the arrays `evaluate_system_control.py`'s own
`compute_metrics()` already has in scope (`times`, `touch_x`, `touch_y`, and the
`segments` list `segment_by_target()` already returns) and returns a dict in the same
style/naming convention as that function's other Extended Metrics (`Average_*_<unit>`,
`*_Trials_Excluded`, `*_Caveat`) -- whoever owns that file can wire it in with
`**jerk_metrics_for_segments(times, touch_x, touch_y, segments)` inside the existing
per-segment loop's accumulation and the final return dict, no further design work
needed. Verified against one real session's telemetry
(`data/01_bronze/session_jetson_track4_20260915_151627/telemetry.csv`) in
`tests/test_trajectory_smoothness.py` -- a real computed number is reported there, not a
placeholder.

Numerical method: three successive `np.gradient(..., edge_order=2)` passes (velocity,
then acceleration, then jerk) against each segment's REAL, non-uniform sample
timestamps -- the same "use the actual irregular dt, not an assumed fixed one" principle
`evaluate_system_control.py`'s own `_integral_error_indices()` already uses for
trapezoidal integration, since frame timing on this system is not perfectly periodic.
`edge_order=2` (not numpy's default of 1) matters a lot here: a quick numerical check
during development showed the default compounding three one-sided first-order
differences at a short segment's boundaries into jerk errors over 10x the interior
(true) value, while `edge_order=2`'s better-conditioned one-sided estimate keeps edge
error within ~5% of the true value even on a short (60-sample) synthetic test -- real
trial segments are typically hundreds of samples, which dilutes any remaining edge
contamination further still.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# Three successive derivatives (velocity -> acceleration -> jerk) need enough interior
# points left that the result is a real signal, not dominated by one-sided edge-difference
# artifacts -- see module docstring's numerical-method note.
MIN_SAMPLES_FOR_JERK: int = 6


def trajectory_jerk_mm_s3(
    seg_times_ms: np.ndarray, seg_touch_x_mm: np.ndarray, seg_touch_y_mm: np.ndarray
) -> Optional[tuple[float, float]]:
    """Ball-trajectory jerk (mm/s^3) over one trial segment. Returns
    `(mean_abs_jerk, rms_jerk)`, or `None` if the segment is too short
    (`< MIN_SAMPLES_FOR_JERK`) or has duplicate/non-monotonic timestamps (which would make
    the derivative chain divide by a near-zero or negative dt -- a real possibility this
    project's telemetry has already shown, e.g. via the known touch-plate glitch pattern;
    `evaluate_system_control.py`'s own `--filter-touch-outliers` is the recommended way to
    clean that up before calling this, not a filter re-implemented here)."""
    n = len(seg_times_ms)
    if n < MIN_SAMPLES_FOR_JERK:
        return None
    t_s = (np.asarray(seg_times_ms, dtype=np.float64) - seg_times_ms[0]) / 1000.0
    if np.any(np.diff(t_s) <= 0.0):
        return None

    x = np.asarray(seg_touch_x_mm, dtype=np.float64)
    y = np.asarray(seg_touch_y_mm, dtype=np.float64)
    vx, vy = np.gradient(x, t_s, edge_order=2), np.gradient(y, t_s, edge_order=2)
    ax, ay = np.gradient(vx, t_s, edge_order=2), np.gradient(vy, t_s, edge_order=2)
    jx, jy = np.gradient(ax, t_s, edge_order=2), np.gradient(ay, t_s, edge_order=2)

    jerk_mag = np.sqrt(jx ** 2 + jy ** 2)
    return float(np.mean(jerk_mag)), float(np.sqrt(np.mean(jerk_mag ** 2)))


def jerk_metrics_for_segments(
    times_ms: np.ndarray,
    touch_x_mm: np.ndarray,
    touch_y_mm: np.ndarray,
    segments: list[tuple[int, int]],
) -> dict[str, object]:
    """Same shape as `evaluate_system_control.py`'s other Extended Metrics: averages
    `trajectory_jerk_mm_s3()` over every segment it could be computed for, and reports how
    many trials were excluded (too short, or a non-monotonic timestamp run) rather than
    silently dropping them. `segments` is the `(start, end)` index-pair list
    `evaluate_system_control.segment_by_target()` already returns -- not recomputed here,
    same inputs the caller's own `compute_metrics()` loop already has."""
    times = np.asarray(times_ms, dtype=np.float64)
    touch_x = np.asarray(touch_x_mm, dtype=np.float64)
    touch_y = np.asarray(touch_y_mm, dtype=np.float64)

    mean_abs_list: list[float] = []
    rms_list: list[float] = []
    for start, end in segments:
        result = trajectory_jerk_mm_s3(times[start:end], touch_x[start:end], touch_y[start:end])
        if result is not None:
            mean_abs_list.append(result[0])
            rms_list.append(result[1])

    total_trials = len(segments)
    return {
        "Average_Jerk_Mean_mm_s3": float(np.mean(mean_abs_list)) if mean_abs_list else None,
        "Average_Jerk_RMS_mm_s3": float(np.mean(rms_list)) if rms_list else None,
        "Jerk_Trials_Excluded": total_trials - len(mean_abs_list),
        "Jerk_Caveat": (
            "Ball-trajectory jerk (3rd time-derivative of touch_x/touch_y, mm/s^3), "
            "distinct from Control_Effort (motor-ANGLE based, not time-normalized). "
            "Computed over each trial's full duration via three successive numerical-"
            "gradient passes against real (non-uniform) sample timestamps. Sensitive to "
            "touch-plate sensor noise/glitches -- consider running with "
            "--filter-touch-outliers for a cleaner signal. Trials with fewer than "
            f"{MIN_SAMPLES_FOR_JERK} samples, or a non-monotonic/duplicate timestamp run, "
            "are excluded (too short/invalid for a 3rd derivative to be meaningful rather "
            "than a numerical artifact)."
        ),
    }
