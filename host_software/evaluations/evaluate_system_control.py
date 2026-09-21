"""
System-level control evaluator. Implements the four standard metrics from
docs/EVALUATION_STRATEGY.md (Steady-State Error, Settling Time, Control Effort,
Task Success Rate) against telemetry CSVs from ANY of the system's controllers:
PID baseline, the expert pipeline (run_eval_expert.py), or a VLA policy
(run_eval_baseline_vla.py / run_eval_our_vla.py).

Those CSVs do not share one timestamp column name (host_timestamp_ms vs.
host_command_sent_ms/host_packet_received_ms vs. the older host_time_ms) --
load_telemetry() normalizes whichever is present. target_x/y, touch_x/y, and
theta_a/b/c are the one column set common to all of them.

Also accepts host_software/src/touch_logger.py's own native CSV schema directly
(confirmed 2026-09-15 while wiring up Track 1 evaluation on the Jetson) -- its
column names differ (target_x_mm/touch_x_mm, host_recv_ts, motor_a/b/c raw steps
instead of theta_a/b/c degrees) since that schema is shared with
estimate_kalman_noise_params.py and wasn't written with this evaluator's column
names in mind. Rather than requiring a separate conversion script or changing
touch_logger.py's schema (which would ripple into the Kalman-fitting workflow),
load_telemetry() aliases the position columns and derives theta_a/b/c from
motor_a/b/c using the same steps-per-degree constant firmware itself uses
(MotorControl.cpp's steps_to_angle(), 3200 steps/revolution) -- no firmware
change needed, since TouchProbe.cpp already sends the raw step counts today.
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
from typing import Any, Dict, List

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_HOST_SOFTWARE_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _HOST_SOFTWARE_DIR not in sys.path:
    sys.path.append(_HOST_SOFTWARE_DIR)

from src.touch_ground_truth_filter import flag_touch_position_outliers

REQUIRED_COLUMNS = [
    "target_x",
    "target_y",
    "touch_x",
    "touch_y",
    "theta_a",
    "theta_b",
    "theta_c",
]

# Priority-ordered: first match wins. Different eval scripts/pipeline
# generations have used different names for "when this sample was received".
TIMESTAMP_CANDIDATES = [
    "host_timestamp_ms",
    "host_packet_received_ms",
    "host_command_sent_ms",
    "host_time_ms",
]

# touch_logger.py's CSV_FIELDS names for the same quantities -- see module
# docstring. Only renamed when the target name isn't already present, so a CSV
# that already uses this evaluator's own naming is untouched.
_TOUCH_LOGGER_RENAMES = {
    "target_x_mm": "target_x",
    "target_y_mm": "target_y",
    "touch_x_mm": "touch_x",
    "touch_y_mm": "touch_y",
}

# MotorControl.cpp's steps_to_angle(): (360.0 / 3200.0) * steps -- 3200 steps/revolution,
# a stable hardware constant (confirmed 2026-09-15 in firmware/stm32_ml_control_and_vision/
# BallBalancingBot/MotorControl.cpp). Inverted here since we're going steps -> degrees.
# Also independently ported in host_software/ml_jetson_vla/runtime/motor_geometry.py
# (DEG_PER_STEP) for Track 4's telemetry-to-training-schema conversion -- not imported
# from here deliberately, since that module lives under one specific arm's runtime and
# this evaluator is arm-agnostic by design (see module docstring). Both are independent
# ports of the same firmware source, not a copy of each other -- if the firmware constant
# (3200 steps/rev) ever changes, update both.
_STEPS_PER_DEGREE = 3200.0 / 360.0

TARGET_CHANGE_TOLERANCE_MM = 1.0  # ignore target jitter below this when segmenting trials
SETTLE_TOLERANCE_MM = 20.0  # from docs/EVALUATION_STRATEGY.md
SETTLE_DURATION_MS = 500.0  # from docs/EVALUATION_STRATEGY.md

# Confirmed 2026-09-15 on 3 real Jetson runs: TARGET_CHANGE_TOLERANCE_MM alone can't
# distinguish a real command transition from a brief marker-classifier misclassification
# burst -- state_machine.py's get_target_coords() only averages marker_history (a plain
# 10-sample rolling mean, no outlier/jump rejection like PredictionGate has for the ball),
# so a sustained ~0.5-1s misclassification (confirmed concentrated in go_green/go_yellow
# phases) rides straight through the average and can reach 29.6mm -- as large as some
# genuine transitions, so raising the mm tolerance alone would still misfire. What
# actually separates noise from a real command in the data is DURATION: every genuine
# transition in docs/EVALUATION_STRATEGY.md's sequence holds ~10,000ms; every observed
# spurious one collapsed within a fraction of a second. 1500ms sits comfortably above the
# observed noise-burst durations and well below the real 10s cadence.
MIN_TRIAL_DURATION_MS = 1500.0


def load_telemetry(csv_path, filter_touch_outliers=False):
    df = pd.read_csv(csv_path)

    df = df.rename(columns={
        src: dst for src, dst in _TOUCH_LOGGER_RENAMES.items()
        if src in df.columns and dst not in df.columns
    })

    # theta_a/b/c aren't in touch_logger.py's schema -- only raw motor_a/b/c step
    # counts are (TouchProbe.cpp already sends these today, no firmware change
    # needed). Derive degrees host-side rather than requiring a separate conversion
    # pass over the CSV.
    for axis in ("a", "b", "c"):
        theta_col, motor_col = f"theta_{axis}", f"motor_{axis}"
        if theta_col not in df.columns and motor_col in df.columns:
            df[theta_col] = df[motor_col] / _STEPS_PER_DEGREE

    if "host_recv_ts" in df.columns and not any(c in df.columns for c in TIMESTAMP_CANDIDATES):
        # touch_logger.py's own timestamp is a time.perf_counter() float in seconds
        # (monotonic, arbitrary zero point) -- fine here since every metric this
        # evaluator computes (settling time, trial segmentation) only uses elapsed
        # deltas within one run, never wall-clock time.
        df["host_timestamp_ms"] = df["host_recv_ts"] * 1000.0

    ts_col = next((c for c in TIMESTAMP_CANDIDATES if c in df.columns), None)
    if ts_col is None:
        raise ValueError(
            f"{csv_path}: no recognized timestamp column (looked for "
            f"{TIMESTAMP_CANDIDATES}, found {list(df.columns)}). Add the new "
            f"column name to TIMESTAMP_CANDIDATES if this is a new schema."
        )
    df = df.rename(columns={ts_col: "timestamp_ms"})

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path}: missing required columns {missing}")

    df = df.dropna(subset=["timestamp_ms"] + REQUIRED_COLUMNS).reset_index(drop=True)

    n_touch_outliers_filtered = 0
    if filter_touch_outliers:
        touch_valid = df["touch_valid"] if "touch_valid" in df.columns else None
        is_outlier = flag_touch_position_outliers(df["touch_x"], df["touch_y"], touch_valid=touch_valid)
        n_touch_outliers_filtered = int(is_outlier.sum())
        df = df.loc[~is_outlier].reset_index(drop=True)

    df["error_mm"] = np.sqrt(
        (df["touch_x"] - df["target_x"]) ** 2 + (df["touch_y"] - df["target_y"]) ** 2
    )
    df.attrs["n_touch_outliers_filtered"] = n_touch_outliers_filtered
    return df


def segment_by_target(df, tol_mm=TARGET_CHANGE_TOLERANCE_MM, min_duration_ms=MIN_TRIAL_DURATION_MS):
    """Split a run into trials wherever target_x/y actually jumps, ignoring
    float noise below tol_mm (a naive exact-equality diff over-segments any
    run where the target isn't a perfectly quantized step signal -- this
    previously produced thousands of spurious 1-row "trials" on real data).

    Then merges any resulting segment shorter than min_duration_ms into its
    preceding segment -- see MIN_TRIAL_DURATION_MS's comment for why a mm-jump
    threshold alone can't reliably tell a real command transition apart from a
    brief marker-classifier misclassification burst."""
    n = len(df)
    if n == 0:
        return []

    changed = np.zeros(n, dtype=bool)
    changed[0] = True
    if n > 1:
        dx = np.abs(np.diff(df["target_x"].to_numpy()))
        dy = np.abs(np.diff(df["target_y"].to_numpy()))
        changed[1:] = (dx > tol_mm) | (dy > tol_mm)

    starts = np.flatnonzero(changed).tolist()
    starts.append(n)
    raw_segments = [(starts[i], starts[i + 1]) for i in range(len(starts) - 1)]

    if min_duration_ms <= 0 or len(raw_segments) <= 1:
        return raw_segments

    times = df["timestamp_ms"].to_numpy(dtype=float)
    merged = [raw_segments[0]]
    for start, end in raw_segments[1:]:
        duration_ms = times[end - 1] - times[start]
        if duration_ms < min_duration_ms:
            # Too brief to be a real, deliberately-held target -- fold into the
            # segment that was actually still active (noise riding on top of it),
            # not counted as its own trial. Chains of consecutive short segments
            # all absorb into the same growing merged[-1] entry here.
            prev_start, _prev_end = merged[-1]
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))

    # The first segment has no preceding one to absorb into (commands can fire in
    # quick succession right at a run's start) -- merge it FORWARD instead. By this
    # point every other entry in `merged` is already guaranteed >= min_duration_ms,
    # so a single check here (not a recursive one) is sufficient.
    if len(merged) > 1:
        first_start, first_end = merged[0]
        if times[first_end - 1] - times[first_start] < min_duration_ms:
            _second_start, second_end = merged[1]
            merged[0:2] = [(first_start, second_end)]

    return merged


def find_settling_time_ms(times_ms, errors_mm, tolerance_mm, duration_ms):
    """First timestamp (relative to times_ms[0]) after which errors_mm stays
    under tolerance_mm for a sustained duration_ms window. Time-based, not
    frame-count-based, so it's correct regardless of sample rate."""
    n = len(times_ms)
    if n == 0:
        return None

    for j in range(n):
        window_end = times_ms[j] + duration_ms
        if window_end > times_ms[-1]:
            # Not enough trailing data to confirm a sustained settle from
            # here on; later j only has even less room, so stop.
            break
        k = int(np.searchsorted(times_ms, window_end, side="right"))
        if np.all(errors_mm[j:k] < tolerance_mm):
            return float(times_ms[j] - times_ms[0])
    return None


def _integral_error_indices(seg_times_ms, seg_errors_mm):
    """IAE/ISE/ITAE (docs/EVALUATION_STRATEGY.md's "Integral Error Indices"
    section) over one trial segment, in mm*s / mm^2*s / mm*s^2 respectively.
    Trapezoidal integration over the segment's actual (jittery) sample
    timestamps, not a fixed dt -- correct regardless of sample rate. Time is
    measured from the segment's own start (t=0 at the command edge), matching
    ITAE's standard definition (penalizes error that persists LATE in the
    transient more than error present at t=0, which is unavoidable). Computed
    over the FULL segment, not just until settle -- unlike Steady-State
    Error/Settling Time, these don't require the trial to have succeeded, so
    every trial contributes a value."""
    t_s = (seg_times_ms - seg_times_ms[0]) / 1000.0
    iae = float(np.trapezoid(np.abs(seg_errors_mm), t_s))
    ise = float(np.trapezoid(seg_errors_mm ** 2, t_s))
    itae = float(np.trapezoid(t_s * np.abs(seg_errors_mm), t_s))
    return iae, ise, itae


def _rise_time_ms(seg_times_ms, seg_errors_mm):
    """Time for the scalar Euclidean error to first drop from its initial
    (segment-start) value E0 to 10% of E0 -- a generalization of the classical
    10%-90% rise time to a decaying (not rising) response, since "distance to
    target" only ever decreases toward a converged value here, never rises the
    way a step response's output does. Returns None if E0 is degenerate
    (<1mm -- already at the target when the command fired, so "rise" isn't a
    meaningful concept for this trial) or the error never reaches the 10%
    threshold within the segment."""
    e0 = seg_errors_mm[0]
    if e0 < 1.0:
        return None
    threshold = 0.1 * e0
    below = np.flatnonzero(seg_errors_mm <= threshold)
    if below.size == 0:
        return None
    return float(seg_times_ms[below[0]] - seg_times_ms[0])


def _axis_overshoot(seg_target, seg_touch):
    """Classical overshoot, computed per-axis (signed error, unlike the
    scalar Euclidean error used elsewhere) since "overshoot" requires a
    direction to overshoot past -- a magnitude-only distance can't go
    negative. e0 = signed error at segment start (touch - target). If the
    signed error ever crosses zero (the ball passes through the target along
    this axis) and swings to the opposite sign, returns
    (peak_opposite_excursion_mm, percent_of_|e0|). Returns (0.0, 0.0) if no
    crossing occurs (no overshoot observed) or e0 is degenerate (<1mm --
    percent would be a division-by-a-near-zero artifact, not a meaningful
    measurement; confirmed empirically on real data, e.g. a 1.1mm initial
    error producing a 462% "overshoot" from a perfectly ordinary ~5mm
    oscillation -- see Overshoot_Caveat)."""
    e0 = seg_touch[0] - seg_target[0]
    if abs(e0) < 1.0:
        return 0.0, 0.0
    err = seg_touch - seg_target
    sign0 = np.sign(e0)
    opposite = (err * sign0) < 0  # crossed to the other side of the target
    if not np.any(opposite):
        return 0.0, 0.0
    peak_opposite = float(np.max(np.abs(err[opposite])))
    return peak_opposite, 100.0 * peak_opposite / abs(e0)


def compute_metrics(df, run_label="run"):
    times = df["timestamp_ms"].to_numpy(dtype=float)
    errors = df["error_mm"].to_numpy(dtype=float)
    target_x = df["target_x"].to_numpy(dtype=float)
    target_y = df["target_y"].to_numpy(dtype=float)
    touch_x = df["touch_x"].to_numpy(dtype=float)
    touch_y = df["touch_y"].to_numpy(dtype=float)

    diff_a = df["theta_a"].diff().abs().dropna()
    diff_b = df["theta_b"].diff().abs().dropna()
    diff_c = df["theta_c"].diff().abs().dropna()
    control_effort_total = float((diff_a + diff_b + diff_c).sum())
    control_effort_per_sample = float((diff_a + diff_b + diff_c).mean())

    segments = segment_by_target(df)
    settling_times, settled_state_errors, successes = [], [], 0
    rise_times, overshoots_mm, overshoots_pct, iae_list, ise_list, itae_list = [], [], [], [], [], []

    for start, end in segments:
        seg_times = times[start:end]
        seg_errors = errors[start:end]
        if len(seg_times) < 2:
            continue  # too short to evaluate settling within

        # Computed for every long-enough trial regardless of success/failure --
        # unlike Steady-State Error/Settling Time below, these don't require a
        # settle to have occurred.
        iae, ise, itae = _integral_error_indices(seg_times, seg_errors)
        iae_list.append(iae)
        ise_list.append(ise)
        itae_list.append(itae)
        rt = _rise_time_ms(seg_times, seg_errors)
        if rt is not None:
            rise_times.append(rt)
        ov_x_mm, ov_x_pct = _axis_overshoot(target_x[start:end], touch_x[start:end])
        ov_y_mm, ov_y_pct = _axis_overshoot(target_y[start:end], touch_y[start:end])
        overshoots_mm.append(max(ov_x_mm, ov_y_mm))
        overshoots_pct.append(max(ov_x_pct, ov_y_pct))

        settle_ms = find_settling_time_ms(
            seg_times, seg_errors, SETTLE_TOLERANCE_MM, SETTLE_DURATION_MS
        )
        if settle_ms is None:
            continue  # never settled -> counts against success rate, excluded from SSE average

        successes += 1
        settling_times.append(settle_ms)
        settle_idx = int(np.searchsorted(seg_times, seg_times[0] + settle_ms, side="left"))
        settled_state_errors.append(float(np.mean(seg_errors[settle_idx:])))

    total_trials = len(segments)
    task_success_rate = (successes / total_trials) * 100.0 if total_trials else 0.0

    return {
        "Run": run_label,
        "Total_Trials": total_trials,
        "Task_Success_Rate_Percent": task_success_rate,
        "Task_Success_Rate_Caveat": (
            "Measures 'settled in time' only -- no ball-drop signal exists in "
            "current telemetry, so a dropped ball is NOT distinguished from a "
            "trial that simply never settled."
        ),
        "Steady_State_Error_mm": (
            float(np.mean(settled_state_errors)) if settled_state_errors else None
        ),
        "Steady_State_Error_Trials_Excluded": total_trials - len(settled_state_errors),
        "Average_Settling_Time_ms": (
            float(np.mean(settling_times)) if settling_times else None
        ),
        "Control_Effort_Per_Sample_deg": control_effort_per_sample,
        "Total_Control_Effort_deg": control_effort_total,
        "Average_Rise_Time_ms": float(np.mean(rise_times)) if rise_times else None,
        "Rise_Time_Trials_Excluded": total_trials - len(rise_times),
        "Rise_Time_Caveat": (
            "Time for scalar error to drop from its trial-start value to 10% of "
            "that value -- a decaying-response generalization of classical rise "
            "time, not the classical 10%-90% rising-step definition. Excluded "
            "trials either started within 1mm of target (nothing to rise from) "
            "or never reached the 10% threshold."
        ),
        "Average_Overshoot_mm": float(np.mean(overshoots_mm)) if overshoots_mm else None,
        "Average_Overshoot_Percent": float(np.mean(overshoots_pct)) if overshoots_pct else None,
        "Overshoot_Caveat": (
            "Per-axis (X, Y) signed-error overshoot past the target; max(X, Y) "
            "reported per trial, then averaged. 0 for a trial with no "
            "zero-crossing (the ball never passed through the target on either "
            "axis) or a degenerate (<1mm) initial error on both axes. "
            "Prefer Average_Overshoot_mm over the _Percent version: percent is "
            "normalized by each trial's own initial error, so a trial that "
            "starts only ~1mm from target can show a triple-digit percent "
            "overshoot from perfectly ordinary millimeter-scale oscillation -- "
            "confirmed on real data, not a hypothetical edge case."
        ),
        "Average_IAE_mm_s": float(np.mean(iae_list)) if iae_list else None,
        "Average_ISE_mm2_s": float(np.mean(ise_list)) if ise_list else None,
        "Average_ITAE_mm_s2": float(np.mean(itae_list)) if itae_list else None,
        "Integral_Error_Indices_Caveat": (
            "IAE/ISE/ITAE integrated over each trial's FULL duration (not just "
            "until settle), so every trial with >=2 samples contributes -- "
            "unlike Steady-State Error/Settling Time, these don't require the "
            "trial to have succeeded."
        ),
        **_inference_latency_stats(df),
    }


def _inference_latency_stats(df):
    """Mean/median/p95/max of vision_inference_ms, if the telemetry has it --
    added 2026-09-18 alongside the touch_ground_truth_filter.py work, not
    present in any CSV recorded before that date. Absent (not zero) on older
    telemetry, so it's never silently averaged in as 0ms."""
    if "vision_inference_ms" not in df.columns:
        return {}
    values = df["vision_inference_ms"].dropna()
    if values.empty:
        return {}
    return {
        "Mean_Inference_Time_ms": float(values.mean()),
        "Median_Inference_Time_ms": float(values.median()),
        "P95_Inference_Time_ms": float(np.percentile(values, 95)),
        "Max_Inference_Time_ms": float(values.max()),
    }


def plot_trajectory(df, output_path, max_rows=None):
    # Confirmed 2026-09-15: the old default (1000) silently truncated a full
    # ~100s/~2450-row Standardized Evaluation Sequence run to its first ~40s,
    # cutting off go_red/HOLD/go_black/RIGHT/BACKWARD/STOP entirely with no
    # warning. None (default) plots everything -- still overridable for a run
    # long enough that plotting every row would actually be unwieldy (e.g. a
    # multi-hour PID baseline log).
    plot_df = df if max_rows is None else df.head(max_rows)
    t0 = plot_df["timestamp_ms"].iloc[0]
    plt.figure(figsize=(10, 5))
    plt.plot(plot_df["timestamp_ms"] - t0, plot_df["target_x"], "r--", label="Target X")
    plt.plot(plot_df["timestamp_ms"] - t0, plot_df["touch_x"], "r-", label="Ball X")
    plt.plot(plot_df["timestamp_ms"] - t0, plot_df["target_y"], "b--", label="Target Y")
    plt.plot(plot_df["timestamp_ms"] - t0, plot_df["touch_y"], "b-", label="Ball Y")
    plt.xlabel("Time (ms)")
    plt.ylabel("Position (mm)")
    plt.title("System Control Trajectory (X/Y)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


_COMPARISON_COLORS: List[str] = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


def _plot_inference_panel(ax: "plt.Axes", all_metrics: List[Dict[str, Any]], labels: List[str]) -> None:
    """Bar = median inference time, upper whisker = P95. Median/P95 rather than
    mean/max because a single scheduling hiccup or warm-up transient moves max
    (and the mean) far more than it should for a like-for-like comparison. A run
    whose CSV predates vision_inference_ms (2026-09-18) has no such keys -- drawn
    as an explicit "no data" bar, never as a real 0ms measurement."""
    medians: List[float] = []
    p95_above_median: List[float] = []
    missing: List[bool] = []
    for m in all_metrics:
        med = m.get("Median_Inference_Time_ms")
        p95 = m.get("P95_Inference_Time_ms")
        if med is None or p95 is None:
            medians.append(0.0)
            p95_above_median.append(0.0)
            missing.append(True)
        else:
            medians.append(float(med))
            p95_above_median.append(max(float(p95) - float(med), 0.0))
            missing.append(False)

    ax.bar(
        labels,
        medians,
        yerr=[[0.0] * len(labels), p95_above_median],
        capsize=4,
        color=_COMPARISON_COLORS[: len(labels)],
    )
    for i, is_missing in enumerate(missing):
        if is_missing:
            ax.text(i, 0, "no data", ha="center", va="bottom", rotation=90, fontsize=8, color="gray")
    ax.set_title("Inference Time (ms)\n(bar = median, whisker = P95, lower is better)")
    ax.tick_params(axis="x", rotation=20)


def plot_comparison(all_metrics: List[Dict[str, Any]], output_path: str) -> None:
    metric_keys = [
        ("Steady_State_Error_mm", "Steady-State Error (mm)", "lower is better"),
        ("Average_Settling_Time_ms", "Settling Time (ms)", "lower is better"),
        ("Control_Effort_Per_Sample_deg", "Control Effort / sample (deg)", "lower is better"),
        ("Task_Success_Rate_Percent", "Task Success Rate (%)", "higher is better"),
    ]
    labels = [m["Run"] for m in all_metrics]

    # Fifth panel only when at least one run actually recorded inference time --
    # comparing runs that all predate vision_inference_ms keeps the original 2x2
    # layout instead of adding an empty panel.
    has_inference = any("Median_Inference_Time_ms" in m for m in all_metrics)
    n_panels = len(metric_keys) + (1 if has_inference else 0)
    ncols = 3 if has_inference else 2

    fig, axes = plt.subplots(2, ncols, figsize=(15 if has_inference else 11, 8))
    flat_axes = list(axes.flat)
    for ax, (key, title, note) in zip(flat_axes, metric_keys):
        values = [m[key] if m[key] is not None else 0.0 for m in all_metrics]
        ax.bar(labels, values, color=_COMPARISON_COLORS[: len(labels)])
        ax.set_title(f"{title}\n({note})")
        ax.tick_params(axis="x", rotation=20)
    if has_inference:
        _plot_inference_panel(flat_axes[len(metric_keys)], all_metrics, labels)
    for unused_ax in flat_axes[n_panels:]:
        unused_ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def evaluate_single_run(csv_path, output_dir, label=None, filter_touch_outliers=False):
    label = label or os.path.splitext(os.path.basename(csv_path))[0]
    print(f"Evaluating telemetry from: {csv_path}")
    df = load_telemetry(csv_path, filter_touch_outliers=filter_touch_outliers)
    if filter_touch_outliers:
        n_flagged = df.attrs.get("n_touch_outliers_filtered", 0)
        print(f"Despiked {n_flagged} touch-plate ground-truth outlier frames "
              f"(see src/touch_ground_truth_filter.py); {len(df)} frames remain.")
    metrics = compute_metrics(df, run_label=label)
    metrics["Touch_Outliers_Filtered"] = df.attrs.get("n_touch_outliers_filtered", 0)

    print(f"\n--- System Control Evaluation: {label} ---")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    suffix = "_filtered" if filter_touch_outliers else ""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"control_metrics{suffix}.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    plot_trajectory(df, os.path.join(output_dir, f"trajectory_plot{suffix}.png"))
    return metrics


def compare_runs(run_specs, output_dir, filter_touch_outliers=False):
    """run_specs: dict of {label: csv_path}."""
    all_metrics = []
    for label, csv_path in run_specs.items():
        df = load_telemetry(csv_path, filter_touch_outliers=filter_touch_outliers)
        metrics = compute_metrics(df, run_label=label)
        metrics["Touch_Outliers_Filtered"] = df.attrs.get("n_touch_outliers_filtered", 0)
        all_metrics.append(metrics)

    table = pd.DataFrame(all_metrics)
    print("\n--- Expert vs. VLA Comparison ---")
    print(table.to_string(index=False))

    suffix = "_filtered" if filter_touch_outliers else ""
    os.makedirs(output_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    json_path = os.path.join(output_dir, f"comparison{suffix}_{stamp}.json")
    csv_path_out = os.path.join(output_dir, f"comparison{suffix}_{stamp}.csv")
    png_path = os.path.join(output_dir, f"comparison{suffix}_{stamp}.png")

    with open(json_path, "w") as f:
        json.dump(all_metrics, f, indent=4)
    table.to_csv(csv_path_out, index=False)
    plot_comparison(all_metrics, png_path)

    print(f"\nSaved comparison to {json_path}, {csv_path_out}, {png_path}")
    return table


def _parse_run_arg(run_arg):
    if "=" not in run_arg:
        raise argparse.ArgumentTypeError(
            f"--runs entries must be label=path/to.csv, got: {run_arg}"
        )
    label, path = run_arg.split("=", 1)
    return label, path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_path", type=str, help="Single-run mode: path to a telemetry CSV"
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        type=_parse_run_arg,
        help="Comparison mode: one or more label=path/to.csv (e.g. "
        "expert=data/04_evaluation/expert_evaluation_run_X.csv "
        "our_vla=data/04_evaluation/labels_sequential_our_vla.csv)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save metrics (single-run mode)",
    )
    parser.add_argument(
        "--report_dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports"),
        help="Directory to save comparison report (comparison mode)",
    )
    parser.add_argument(
        "--filter-touch-outliers", action="store_true",
        help="Despike touch-plate ground-truth readings (see "
        "src/touch_ground_truth_filter.py) before computing metrics. Off by "
        "default so existing results are never silently changed -- outputs are "
        "written to separate _filtered-suffixed files, not in place.",
    )
    args = parser.parse_args()

    if args.runs:
        compare_runs(dict(args.runs), args.report_dir, filter_touch_outliers=args.filter_touch_outliers)
    elif args.csv_path:
        evaluate_single_run(args.csv_path, args.output_dir, filter_touch_outliers=args.filter_touch_outliers)
    else:
        parser.error("Provide either --csv_path (single run) or --runs (comparison)")
