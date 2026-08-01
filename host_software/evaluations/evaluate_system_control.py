import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse


def evaluate_telemetry(csv_path, output_dir):
    print(f"Evaluating telemetry from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Need to handle dropping NA or missing rows if any
    df = df.dropna(
        subset=[
            "target_x",
            "target_y",
            "touch_x",
            "touch_y",
            "theta_a",
            "theta_b",
            "theta_c",
        ]
    )

    # 1. Steady-State Error
    # Calculate euclidean error for every frame
    errors = np.sqrt(
        (df["touch_x"] - df["target_x"]) ** 2 + (df["touch_y"] - df["target_y"]) ** 2
    )
    steady_state_error = float(errors.mean())

    # 2. Control Effort (Theta Jerk)
    # Sum of absolute differences in theta between consecutive frames
    diff_a = np.abs(df["theta_a"].diff().dropna())
    diff_b = np.abs(df["theta_b"].diff().dropna())
    diff_c = np.abs(df["theta_c"].diff().dropna())

    control_effort_per_frame = (diff_a + diff_b + diff_c).mean()
    total_control_effort = float((diff_a + diff_b + diff_c).sum())

    # 3. Settling Time & 4. Task Success Rate
    # A task is defined as a period where the target remains constant.
    # We find where target_x or target_y changes.
    df["target_change"] = (df["target_x"].diff() != 0) | (df["target_y"].diff() != 0)

    # Get indices of changes
    change_idx = df.index[df["target_change"]].tolist()

    # If there are no target changes, the whole file is one task
    if len(change_idx) == 1 and change_idx[0] == 0:
        change_idx.append(len(df))
    elif len(change_idx) == 0:
        change_idx = [0, len(df)]
    else:
        change_idx.append(len(df))

    settling_times = []
    successes = 0
    total_tasks = len(change_idx) - 1

    # Assume 20mm is the acceptable tolerance for settling
    TOLERANCE_MM = 20.0
    # Needs to be settled for 10 frames (approx 330ms at 30fps)
    SETTLE_FRAMES = 10

    for i in range(total_tasks):
        start = change_idx[i]
        end = change_idx[i + 1]

        task_df = df.iloc[start:end]
        if len(task_df) < SETTLE_FRAMES:
            continue

        task_errors = errors.iloc[start:end].values
        task_times = task_df["host_timestamp_ms"].values

        start_time = task_times[0]

        # Find when it settles
        settled = False
        for j in range(len(task_errors) - SETTLE_FRAMES):
            if np.all(task_errors[j : j + SETTLE_FRAMES] < TOLERANCE_MM):
                settling_time = task_times[j] - start_time
                settling_times.append(float(settling_time))
                settled = True
                break

        if settled:
            successes += 1

    avg_settling_time = float(np.mean(settling_times)) if settling_times else 0.0
    task_success_rate = (successes / total_tasks) * 100.0 if total_tasks > 0 else 0.0

    metrics = {
        "Total_Trials": total_tasks,
        "Task_Success_Rate_Percent": task_success_rate,
        "Steady_State_Error_mm": steady_state_error,
        "Average_Settling_Time_ms": avg_settling_time,
        "Control_Effort_Per_Frame_deg": float(control_effort_per_frame),
        "Total_Control_Effort_deg": total_control_effort,
    }

    print("\n--- System Control Evaluation ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}")

    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "control_metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=4)

    # Plot Trajectory (First 1000 frames for clarity)
    plot_df = df.head(1000)
    plt.figure(figsize=(10, 5))
    plt.plot(
        plot_df["host_timestamp_ms"] - plot_df["host_timestamp_ms"].iloc[0],
        plot_df["target_x"],
        "r--",
        label="Target X",
    )
    plt.plot(
        plot_df["host_timestamp_ms"] - plot_df["host_timestamp_ms"].iloc[0],
        plot_df["touch_x"],
        "r-",
        label="Ball X",
    )
    plt.plot(
        plot_df["host_timestamp_ms"] - plot_df["host_timestamp_ms"].iloc[0],
        plot_df["target_y"],
        "b--",
        label="Target Y",
    )
    plt.plot(
        plot_df["host_timestamp_ms"] - plot_df["host_timestamp_ms"].iloc[0],
        plot_df["touch_y"],
        "b-",
        label="Ball Y",
    )
    plt.xlabel("Time (ms)")
    plt.ylabel("Position (mm)")
    plt.title("System Control Trajectory (X/Y)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "trajectory_plot.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_path", type=str, required=True, help="Path to telemetry CSV"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results", help="Directory to save metrics"
    )
    args = parser.parse_args()

    evaluate_telemetry(args.csv_path, args.output_dir)
