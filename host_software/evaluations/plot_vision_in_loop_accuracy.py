"""Vision-in-the-loop accuracy analysis: how well the Shared Backbone CNN's live
ball-position estimate tracked ground truth during an actual closed-loop run, as
opposed to evaluate_shared_vision_backbone.py's static held-out-dataset numbers.

Reuses evaluate_system_control.py's load_telemetry() rather than re-implementing
CSV parsing/column normalization -- this script only adds analysis of the vision
columns (vision_x_mm/vision_y_mm, raw_vision_x_mm/raw_vision_y_mm, raw_err_mm/
err_mm) that load_telemetry() already passes through untouched from
src/touch_logger.py's native CSV schema, since evaluate_system_control.py itself
never reads them (its REQUIRED_COLUMNS/metrics are vision-model-agnostic by
design).

Column semantics (see src/touch_logger.py's CSV_FIELDS comments, authoritative):
  - raw_vision_x_mm/raw_vision_y_mm: the CNN's raw inference output, BEFORE the
    gate/dead-band/MLP correction stage. This is "what the model actually saw."
  - vision_x_mm/vision_y_mm: the same estimate AFTER gating/dead-band/MLP -- what
    was actually sent to the STM32 as the control loop's ball-position input.
  - touch_x_mm/touch_y_mm (aliased to touch_x/touch_y by load_telemetry): ground
    truth from the physical touch-plate sensor.
  - raw_err_mm / err_mm: already-computed Euclidean error of raw/processed vision
    against touch-plate ground truth, blank wherever touch_valid==0 (no ball on
    the plate) since there's no ground truth to compare against for that frame.

Usage:
    python plot_vision_in_loop_accuracy.py --csv_path <telemetry.csv> --output_dir <dir>
    python plot_vision_in_loop_accuracy.py --runs run1=<csv1> run2=<csv2> --report_dir <dir>
"""

import argparse
import json
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_system_control import load_telemetry  # noqa: E402

VISION_COLUMNS = [
    "raw_vision_x_mm",
    "raw_vision_y_mm",
    "vision_x_mm",
    "vision_y_mm",
    "raw_err_mm",
    "err_mm",
]


def _require_vision_columns(df, csv_path):
    missing = [c for c in VISION_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path}: missing vision columns {missing} -- this CSV was not "
            "produced by src/touch_logger.py's TouchTelemetryLogger (or raw_x/raw_y "
            "were never passed to send_frame()). Nothing to analyze."
        )


def compute_vision_accuracy_metrics(df, run_label="run"):
    """Summarizes in-the-loop vision accuracy against touch-plate ground truth,
    for both the raw CNN estimate and the post-gate/dead-band/MLP-corrected
    estimate actually used by the control loop. Only rows with a valid ground
    truth reading (raw_err_mm/err_mm non-null) contribute -- matches
    src/touch_logger.py's own convention of leaving these blank rather than 0
    when touch_valid==0, so a mean here is never diluted by absent-ball frames."""
    total_frames = int(len(df))

    raw_valid = df["raw_err_mm"].dropna()
    proc_valid = df["err_mm"].dropna()
    raw_estimate_available = int(df["raw_vision_x_mm"].notna().sum())

    def _stats(series):
        if series.empty:
            return {"mean_mm": None, "median_mm": None, "p95_mm": None, "max_mm": None, "n": 0}
        return {
            "mean_mm": float(series.mean()),
            "median_mm": float(series.median()),
            "p95_mm": float(np.percentile(series, 95)),
            "max_mm": float(series.max()),
            "n": int(len(series)),
        }

    return {
        "Run": run_label,
        "Total_Frames": total_frames,
        "Raw_Vision_Estimate_Available_Frames": raw_estimate_available,
        "Raw_Vision_Estimate_Available_Percent": (
            100.0 * raw_estimate_available / total_frames if total_frames else 0.0
        ),
        "Ground_Truth_Matched_Frames": int(len(raw_valid)),
        "Ground_Truth_Matched_Percent": (
            100.0 * len(raw_valid) / total_frames if total_frames else 0.0
        ),
        "Raw_CNN_Vision_Error_vs_Touch_Plate": _stats(raw_valid),
        "Processed_Vision_Error_vs_Touch_Plate": _stats(proc_valid),
        "Note": (
            "Raw = CNN inference output before gate/dead-band/MLP correction "
            "(what the model actually predicted). Processed = after correction "
            "(what the control loop actually used). Both measured in the live "
            "closed loop against the touch-plate sensor as ground truth -- "
            "distinct from evaluate_shared_vision_backbone.py's static "
            "held-out-dataset numbers."
        ),
    }


def plot_trajectory_with_vision(df, output_path, max_rows=None, title_suffix=""):
    """Three-panel figure: X position, Y position, and resulting Euclidean
    error over time, each showing Target (reference setpoint), Ball
    (touch-plate ground truth), and the CNN's raw vision estimate."""
    plot_df = df if max_rows is None else df.head(max_rows)
    t = (plot_df["timestamp_ms"] - plot_df["timestamp_ms"].iloc[0]).to_numpy()

    fig, (ax_x, ax_y, ax_err) = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

    ax_x.plot(t, plot_df["target_x"], "k--", label="Target X (reference setpoint)", linewidth=1.2)
    ax_x.plot(t, plot_df["touch_x"], "r-", label="Ball X (touch-plate ground truth)", linewidth=1.0)
    ax_x.plot(t, plot_df["raw_vision_x_mm"], color="tab:green", linestyle=":", label="CNN raw vision estimate X", linewidth=1.0)
    ax_x.set_ylabel("X Position (mm)")
    ax_x.set_title(f"System Control Trajectory with Vision-in-the-Loop Estimate{title_suffix}")
    ax_x.legend(loc="upper right", fontsize=8)

    ax_y.plot(t, plot_df["target_y"], "k--", label="Target Y (reference setpoint)", linewidth=1.2)
    ax_y.plot(t, plot_df["touch_y"], "b-", label="Ball Y (touch-plate ground truth)", linewidth=1.0)
    ax_y.plot(t, plot_df["raw_vision_y_mm"], color="tab:green", linestyle=":", label="CNN raw vision estimate Y", linewidth=1.0)
    ax_y.set_ylabel("Y Position (mm)")
    ax_y.legend(loc="upper right", fontsize=8)

    ax_err.plot(t, plot_df["raw_err_mm"], color="tab:green", linewidth=0.8, label="Raw CNN error vs. touch-plate (mm)")
    ax_err.plot(t, plot_df["err_mm"], color="tab:purple", linewidth=0.8, alpha=0.7, label="Processed vision error vs. touch-plate (mm)")
    ax_err.set_ylabel("Euclidean Error (mm)")
    ax_err.set_xlabel("Time (ms)")
    ax_err.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_accuracy_comparison(all_metrics, output_path):
    labels = [m["Run"] for m in all_metrics]
    raw_means = [m["Raw_CNN_Vision_Error_vs_Touch_Plate"]["mean_mm"] or 0.0 for m in all_metrics]
    raw_p95s = [m["Raw_CNN_Vision_Error_vs_Touch_Plate"]["p95_mm"] or 0.0 for m in all_metrics]
    proc_means = [m["Processed_Vision_Error_vs_Touch_Plate"]["mean_mm"] or 0.0 for m in all_metrics]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    width = 0.35
    x = np.arange(len(labels))
    ax1.bar(x - width / 2, raw_means, width, label="Raw CNN estimate", color="tab:green")
    ax1.bar(x + width / 2, proc_means, width, label="Processed (post-gate/MLP)", color="tab:purple")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20)
    ax1.set_ylabel("Mean Euclidean Error vs. Touch Plate (mm)")
    ax1.set_title("In-the-Loop Vision Accuracy\n(mean, lower is better)")
    ax1.legend()

    ax2.bar(x, raw_p95s, color="tab:green")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=20)
    ax2.set_ylabel("95th Percentile Error (mm)")
    ax2.set_title("Raw CNN Estimate: p95 Error vs. Touch Plate\n(lower is better)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def evaluate_single_run(csv_path, output_dir, label=None, max_rows=None, filter_touch_outliers=False):
    label = label or os.path.splitext(os.path.basename(csv_path))[0]
    print(f"Analyzing vision-in-the-loop accuracy from: {csv_path}")
    df = load_telemetry(csv_path, filter_touch_outliers=filter_touch_outliers)
    _require_vision_columns(df, csv_path)
    if filter_touch_outliers:
        n_flagged = df.attrs.get("n_touch_outliers_filtered", 0)
        print(f"Despiked {n_flagged} touch-plate ground-truth outlier frames; "
              f"{len(df)} frames remain.")

    metrics = compute_vision_accuracy_metrics(df, run_label=label)
    metrics["Touch_Outliers_Filtered"] = df.attrs.get("n_touch_outliers_filtered", 0)
    print(f"\n--- Vision-in-the-Loop Accuracy: {label} ---")
    print(json.dumps(metrics, indent=2))

    suffix = "_filtered" if filter_touch_outliers else ""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"vision_accuracy_metrics{suffix}.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    plot_trajectory_with_vision(
        df, os.path.join(output_dir, f"trajectory_with_vision_plot{suffix}.png"), max_rows=max_rows
    )
    return metrics


def compare_runs(run_specs, output_dir, filter_touch_outliers=False):
    all_metrics = []
    for label, csv_path in run_specs.items():
        df = load_telemetry(csv_path, filter_touch_outliers=filter_touch_outliers)
        _require_vision_columns(df, csv_path)
        metrics = compute_vision_accuracy_metrics(df, run_label=label)
        metrics["Touch_Outliers_Filtered"] = df.attrs.get("n_touch_outliers_filtered", 0)
        all_metrics.append(metrics)

    table = pd.json_normalize(all_metrics)
    print("\n--- Vision-in-the-Loop Accuracy Comparison ---")
    print(table.to_string(index=False))

    suffix = "_filtered" if filter_touch_outliers else ""
    os.makedirs(output_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    json_path = os.path.join(output_dir, f"vision_accuracy_comparison{suffix}_{stamp}.json")
    csv_path_out = os.path.join(output_dir, f"vision_accuracy_comparison{suffix}_{stamp}.csv")
    png_path = os.path.join(output_dir, f"vision_accuracy_comparison{suffix}_{stamp}.png")

    with open(json_path, "w") as f:
        json.dump(all_metrics, f, indent=4)
    table.to_csv(csv_path_out, index=False)
    plot_accuracy_comparison(all_metrics, png_path)

    print(f"\nSaved comparison to {json_path}, {csv_path_out}, {png_path}")
    return table


def _parse_run_arg(run_arg):
    if "=" not in run_arg:
        raise argparse.ArgumentTypeError(f"--runs entries must be label=path/to.csv, got: {run_arg}")
    label, path = run_arg.split("=", 1)
    return label, path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv_path", type=str, help="Single-run mode: path to a telemetry CSV")
    parser.add_argument(
        "--runs", nargs="+", type=_parse_run_arg,
        help="Comparison mode: one or more label=path/to.csv",
    )
    parser.add_argument("--output_dir", type=str, default="results", help="Single-run mode output dir")
    parser.add_argument(
        "--report_dir", type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports"),
        help="Comparison mode output dir",
    )
    parser.add_argument(
        "--max_rows", type=int, default=None,
        help="Truncate the single-run trajectory plot to the first N rows (default: all)",
    )
    parser.add_argument(
        "--filter-touch-outliers", action="store_true",
        help="Despike touch-plate ground-truth readings (see "
        "src/touch_ground_truth_filter.py) before computing vision-accuracy "
        "stats. Off by default; outputs go to separate _filtered-suffixed files.",
    )
    args = parser.parse_args()

    if args.runs:
        compare_runs(dict(args.runs), args.report_dir, filter_touch_outliers=args.filter_touch_outliers)
    elif args.csv_path:
        evaluate_single_run(
            args.csv_path, args.output_dir, max_rows=args.max_rows,
            filter_touch_outliers=args.filter_touch_outliers,
        )
    else:
        parser.error("Provide either --csv_path (single run) or --runs (comparison)")
