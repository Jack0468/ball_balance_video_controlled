"""Sweeps the settle-tolerance threshold (the radius around the target a trial
must enter and hold for SETTLE_DURATION_MS to count as "reached the target")
and shows how Task Success Rate trades off against it -- i.e. what happens as
the definition of "on target" is tightened or loosened.

Reuses evaluate_system_control.py's load_telemetry()/segment_by_target()/
find_settling_time_ms() directly rather than re-implementing trial
segmentation or the settling-time search -- only SETTLE_TOLERANCE_MM is
varied; everything else (trial boundaries, duration requirement) matches the
project's standard metric exactly at tolerance=20mm (EVALUATION_STRATEGY.md's
own value).

Usage:
    python plot_success_rate_vs_tolerance.py --runs label=path.csv ... --output_path out.png
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.append(_THIS_DIR)

from evaluate_system_control import (  # noqa: E402
    SETTLE_DURATION_MS,
    SETTLE_TOLERANCE_MM,
    find_settling_time_ms,
    load_telemetry,
    segment_by_target,
)

DEFAULT_TOLERANCES_MM = list(range(2, 51, 2))

# Per EVALUATION_STRATEGY.md: "the vision system's own noise floor is ~4-5mm
# RMSE" -- a tolerance tighter than this asks the system to hold within a
# radius smaller than the ground-truth measurement's own noise, which is not
# a meaningful operating point regardless of true control performance.
VISION_NOISE_FLOOR_MM = 5.0


def compute_success_at_tolerance(df, tolerance_mm, duration_ms=SETTLE_DURATION_MS):
    """Same trial-segmentation and per-trial success logic as
    evaluate_system_control.compute_metrics(), parameterized over tolerance."""
    times = df["timestamp_ms"].to_numpy(dtype=float)
    errors = df["error_mm"].to_numpy(dtype=float)
    segments = segment_by_target(df)
    total_trials = len(segments)
    successes = 0
    settling_times = []

    for start, end in segments:
        seg_times = times[start:end]
        seg_errors = errors[start:end]
        if len(seg_times) < 2:
            continue
        settle_ms = find_settling_time_ms(seg_times, seg_errors, tolerance_mm, duration_ms)
        if settle_ms is None:
            continue
        successes += 1
        settling_times.append(settle_ms)

    return total_trials, successes, settling_times


def sweep(run_csvs, tolerances, filter_touch_outliers=True):
    dfs = {label: load_telemetry(path, filter_touch_outliers=filter_touch_outliers) for label, path in run_csvs.items()}

    per_run = {label: {"success_rate": [], "mean_settle_ms": []} for label in dfs}
    agg_success_rate = []
    agg_mean_settle_ms = []

    for tol in tolerances:
        total_trials_all = 0
        successes_all = 0
        settle_all = []

        for label, df in dfs.items():
            total_trials, successes, settling_times = compute_success_at_tolerance(df, tol)
            total_trials_all += total_trials
            successes_all += successes
            settle_all.extend(settling_times)

            per_run[label]["success_rate"].append(100.0 * successes / total_trials if total_trials else 0.0)
            per_run[label]["mean_settle_ms"].append(float(np.mean(settling_times)) if settling_times else np.nan)

        agg_success_rate.append(100.0 * successes_all / total_trials_all if total_trials_all else 0.0)
        agg_mean_settle_ms.append(float(np.mean(settle_all)) if settle_all else np.nan)

    return agg_success_rate, agg_mean_settle_ms, per_run


def plot(tolerances, agg_success_rate, agg_mean_settle_ms, per_run, output_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(per_run)))
    for (label, data), color in zip(per_run.items(), colors):
        ax1.plot(tolerances, data["success_rate"], alpha=0.5, linewidth=1.2, color=color, label=label)
    ax1.plot(tolerances, agg_success_rate, color="black", linewidth=2.8, label="All runs combined")
    ax1.axvline(SETTLE_TOLERANCE_MM, color="red", linestyle="--", linewidth=1.2,
                label=f"Current threshold ({SETTLE_TOLERANCE_MM:.0f}mm)")
    ax1.axvspan(0, VISION_NOISE_FLOOR_MM, color="grey", alpha=0.15,
                label=f"Below vision noise floor (~{VISION_NOISE_FLOOR_MM:.0f}mm)")
    ax1.set_xlabel("Settle-tolerance threshold (mm) -- stricter to the left")
    ax1.set_ylabel("Task Success Rate (%)")
    ax1.set_title("Success Rate vs. Target-Reach Tolerance")
    ax1.set_ylim(-5, 105)
    ax1.legend(fontsize=7, loc="lower right")
    ax1.grid(True, linestyle=":", alpha=0.5)

    ax2.plot(tolerances, agg_mean_settle_ms, color="black", linewidth=2.8)
    ax2.axvline(SETTLE_TOLERANCE_MM, color="red", linestyle="--", linewidth=1.2)
    ax2.axvspan(0, VISION_NOISE_FLOOR_MM, color="grey", alpha=0.15)
    ax2.set_xlabel("Settle-tolerance threshold (mm) -- stricter to the left")
    ax2.set_ylabel("Mean Settling Time (ms), successful trials only")
    ax2.set_title("Settling Time vs. Target-Reach Tolerance")
    ax2.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _parse_run_arg(run_arg):
    if "=" not in run_arg:
        raise argparse.ArgumentTypeError(f"--runs entries must be label=path/to.csv, got: {run_arg}")
    label, path = run_arg.split("=", 1)
    return label, path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=_parse_run_arg, required=True,
                         help="One or more label=path/to.csv")
    parser.add_argument("--output_path", type=str, default="success_rate_vs_tolerance.png")
    parser.add_argument("--min_tolerance_mm", type=float, default=2.0)
    parser.add_argument("--max_tolerance_mm", type=float, default=50.0)
    parser.add_argument("--step_mm", type=float, default=2.0)
    parser.add_argument("--no-filter-touch-outliers", dest="filter_touch_outliers", action="store_false")
    parser.set_defaults(filter_touch_outliers=True)
    args = parser.parse_args()

    tolerances = list(np.arange(args.min_tolerance_mm, args.max_tolerance_mm + args.step_mm, args.step_mm))

    agg_success_rate, agg_mean_settle_ms, per_run = sweep(
        dict(args.runs), tolerances, filter_touch_outliers=args.filter_touch_outliers
    )

    print("Tolerance(mm)  SuccessRate(%)  MeanSettleMs")
    for tol, sr, ms in zip(tolerances, agg_success_rate, agg_mean_settle_ms):
        print(f"{tol:>10.1f}  {sr:>13.1f}  {ms:>11.1f}" if not np.isnan(ms) else f"{tol:>10.1f}  {sr:>13.1f}  {'--':>11}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)) or ".", exist_ok=True)
    plot(tolerances, agg_success_rate, agg_mean_settle_ms, per_run, args.output_path)
    print(f"\nSaved plot to {args.output_path}")
