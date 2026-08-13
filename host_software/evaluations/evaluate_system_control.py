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
"""

import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse

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

TARGET_CHANGE_TOLERANCE_MM = 1.0  # ignore target jitter below this when segmenting trials
SETTLE_TOLERANCE_MM = 20.0  # from docs/EVALUATION_STRATEGY.md
SETTLE_DURATION_MS = 500.0  # from docs/EVALUATION_STRATEGY.md


def load_telemetry(csv_path):
    df = pd.read_csv(csv_path)

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
    df["error_mm"] = np.sqrt(
        (df["touch_x"] - df["target_x"]) ** 2 + (df["touch_y"] - df["target_y"]) ** 2
    )
    return df


def segment_by_target(df, tol_mm=TARGET_CHANGE_TOLERANCE_MM):
    """Split a run into trials wherever target_x/y actually jumps, ignoring
    float noise below tol_mm (a naive exact-equality diff over-segments any
    run where the target isn't a perfectly quantized step signal -- this
    previously produced thousands of spurious 1-row "trials" on real data)."""
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
    return [(starts[i], starts[i + 1]) for i in range(len(starts) - 1)]


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


def compute_metrics(df, run_label="run"):
    times = df["timestamp_ms"].to_numpy(dtype=float)
    errors = df["error_mm"].to_numpy(dtype=float)

    diff_a = df["theta_a"].diff().abs().dropna()
    diff_b = df["theta_b"].diff().abs().dropna()
    diff_c = df["theta_c"].diff().abs().dropna()
    control_effort_total = float((diff_a + diff_b + diff_c).sum())
    control_effort_per_sample = float((diff_a + diff_b + diff_c).mean())

    segments = segment_by_target(df)
    settling_times, settled_state_errors, successes = [], [], 0

    for start, end in segments:
        seg_times = times[start:end]
        seg_errors = errors[start:end]
        if len(seg_times) < 2:
            continue  # too short to evaluate settling within

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
    }


def plot_trajectory(df, output_path, max_rows=1000):
    plot_df = df.head(max_rows)
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


def plot_comparison(all_metrics, output_path):
    metric_keys = [
        ("Steady_State_Error_mm", "Steady-State Error (mm)", "lower is better"),
        ("Average_Settling_Time_ms", "Settling Time (ms)", "lower is better"),
        ("Control_Effort_Per_Sample_deg", "Control Effort / sample (deg)", "lower is better"),
        ("Task_Success_Rate_Percent", "Task Success Rate (%)", "higher is better"),
    ]
    labels = [m["Run"] for m in all_metrics]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (key, title, note) in zip(axes.flat, metric_keys):
        values = [m[key] if m[key] is not None else 0.0 for m in all_metrics]
        ax.bar(labels, values, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"][: len(labels)])
        ax.set_title(f"{title}\n({note})")
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def evaluate_single_run(csv_path, output_dir, label=None):
    label = label or os.path.splitext(os.path.basename(csv_path))[0]
    print(f"Evaluating telemetry from: {csv_path}")
    df = load_telemetry(csv_path)
    metrics = compute_metrics(df, run_label=label)

    print(f"\n--- System Control Evaluation: {label} ---")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "control_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    plot_trajectory(df, os.path.join(output_dir, "trajectory_plot.png"))
    return metrics


def compare_runs(run_specs, output_dir):
    """run_specs: dict of {label: csv_path}."""
    all_metrics = []
    for label, csv_path in run_specs.items():
        df = load_telemetry(csv_path)
        all_metrics.append(compute_metrics(df, run_label=label))

    table = pd.DataFrame(all_metrics)
    print("\n--- Expert vs. VLA Comparison ---")
    print(table.to_string(index=False))

    os.makedirs(output_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    json_path = os.path.join(output_dir, f"comparison_{stamp}.json")
    csv_path_out = os.path.join(output_dir, f"comparison_{stamp}.csv")
    png_path = os.path.join(output_dir, f"comparison_{stamp}.png")

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
    args = parser.parse_args()

    if args.runs:
        compare_runs(dict(args.runs), args.report_dir)
    elif args.csv_path:
        evaluate_single_run(args.csv_path, args.output_dir)
    else:
        parser.error("Provide either --csv_path (single run) or --runs (comparison)")
