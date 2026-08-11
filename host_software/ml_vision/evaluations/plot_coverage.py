import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_coverage(csv_path):
    print(f"Reading dataset: {csv_path}")

    # Read CSV
    df = pd.read_csv(csv_path)

    # Determine column names
    x_col, y_col = None, None
    if "touch_x" in df.columns and "touch_y" in df.columns:
        x_col, y_col = "touch_x", "touch_y"
    elif "target_x" in df.columns and "target_y" in df.columns:
        x_col, y_col = "target_x", "target_y"
    else:
        print("Error: Could not find touch_x/touch_y or target_x/target_y in CSV.")
        print(f"Available columns: {df.columns}")
        return

    x = df[x_col].values
    y = df[y_col].values

    # Coverage Metric Calculation
    # We define a "unit area" as a 10x10mm grid cell within the safe zone [-80, 80] x [-60, 60]
    safe_x_range = [-80, 80]
    safe_y_range = [-60, 60]
    grid_size_mm = 2.5

    x_bins = int((safe_x_range[1] - safe_x_range[0]) / grid_size_mm)
    y_bins = int((safe_y_range[1] - safe_y_range[0]) / grid_size_mm)

    # Filter points to only those inside the safe zone
    mask = (
        (x >= safe_x_range[0])
        & (x <= safe_x_range[1])
        & (y >= safe_y_range[0])
        & (y <= safe_y_range[1])
    )
    safe_x = x[mask]
    safe_y = y[mask]

    # Calculate 2D histogram for metric
    H, xedges, yedges = np.histogram2d(
        safe_x, safe_y, bins=[x_bins, y_bins], range=[safe_x_range, safe_y_range]
    )

    # Goal: We want a minimum of 1 frame in every 2.5x2.5mm area
    MIN_SAMPLES_PER_UNIT = 1
    total_cells = x_bins * y_bins
    cells_meeting_goal = np.sum(H >= MIN_SAMPLES_PER_UNIT)
    coverage_percentage = (cells_meeting_goal / total_cells) * 100.0

    print("-" * 50)
    print("COVERAGE METRICS:")
    print(
        f"Total safe zone area evaluated: {safe_x_range[0]} to {safe_x_range[1]} (X), {safe_y_range[0]} to {safe_y_range[1]} (Y)"
    )
    print(
        f"Grid size: {grid_size_mm}x{grid_size_mm} mm ({total_cells} total grid cells)"
    )
    print(f"Goal: Minimum {MIN_SAMPLES_PER_UNIT} samples per grid cell")
    print(f"Result: {cells_meeting_goal}/{total_cells} cells met the goal.")
    print(f"Coverage Score: {coverage_percentage:.2f}%")
    print("-" * 50)

    # A single pass/fail threshold says little once most sessions clear it
    # comfortably -- report the full per-cell density distribution too, so
    # sessions can be compared against each other rather than just against 10.
    flat_H = H.flatten()
    empty_cells = int(np.sum(flat_H == 0))
    print("CELL DENSITY DISTRIBUTION (samples per grid cell):")
    print(f"  Empty cells (0 samples): {empty_cells}/{total_cells}")
    print(f"  Min:    {int(flat_H.min())}")
    print(f"  P25:    {np.percentile(flat_H, 25):.1f}")
    print(f"  Median: {np.median(flat_H):.1f}")
    print(f"  Mean:   {flat_H.mean():.1f}")
    print(f"  P75:    {np.percentile(flat_H, 75):.1f}")
    print(f"  Max:    {int(flat_H.max())}")
    print(f"  Std:    {flat_H.std():.1f}")
    print("-" * 50)
    print("COVERAGE AT OTHER THRESHOLDS:")
    dynamic_thresholds = sorted(set(int(round(t)) for t in np.linspace(flat_H.min(), flat_H.max(), 5)))
    for threshold in dynamic_thresholds:
        cells_meeting = int(np.sum(flat_H >= threshold))
        pct = (cells_meeting / total_cells) * 100.0
        print(f"  >= {threshold:>3} samples/cell: {cells_meeting}/{total_cells} cells ({pct:.2f}%)")
    print("-" * 50)

    plt.figure(figsize=(18, 5))

    # 1. Scatter Plot (Trajectory)
    plt.subplot(1, 3, 1)
    plt.scatter(x, y, s=1, alpha=0.3, color="blue")
    plt.xlim(-100, 100)
    plt.ylim(-75, 75)
    # Draw safe zone perimeter
    plt.plot(
        [-80, 80, 80, -80, -80],
        [-60, -60, 60, 60, -60],
        "r--",
        label="Safe Zone Perimeter",
    )
    plt.title("Ball Trajectory (Scatter)")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right")
    plt.gca().set_aspect("equal", adjustable="box")

    # 2. 2D Histogram (Heatmap of Time Spent)
    plt.subplot(1, 3, 2)
    h = plt.hist2d(
        x, y, bins=(80, 60), range=[[-100, 100], [-75, 75]], cmap="inferno", cmin=1
    )
    plt.colorbar(label="Frames spent in area")
    # Draw safe zone perimeter
    plt.plot(
        [-80, 80, 80, -80, -80],
        [-60, -60, 60, 60, -60],
        "w--",
        alpha=0.5,
        label="Safe Zone Perimeter",
    )
    plt.title(
        f"Spatial Coverage Density\nGoal Met: {coverage_percentage:.1f}% (>={MIN_SAMPLES_PER_UNIT} samples per {grid_size_mm}x{grid_size_mm}mm)"
    )
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.legend(loc="upper right")

    # 3. Distribution of per-cell sample counts (how dense is a "typical" cell,
    # not just whether it clears one fixed threshold)
    plt.subplot(1, 3, 3)
    plt.hist(flat_H, bins=30, color="steelblue", edgecolor="black", alpha=0.8)
    plt.axvline(np.median(flat_H), color="green", linestyle="--", label=f"Median: {np.median(flat_H):.0f}")
    plt.axvline(flat_H.mean(), color="orange", linestyle="--", label=f"Mean: {flat_H.mean():.1f}")
    plt.axvline(MIN_SAMPLES_PER_UNIT, color="red", linestyle=":", label=f"Goal: {MIN_SAMPLES_PER_UNIT}")
    plt.title("Per-Cell Sample Count Distribution")
    plt.xlabel("Samples in cell")
    plt.ylabel("Number of cells")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()

    # Save the plot
    output_filename = os.path.splitext(csv_path)[0] + "_coverage_plot.png"
    plt.savefig(output_filename, dpi=300)
    print(f"Saved plot to: {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot the physical board coverage of the ball during a dataset collection run."
    )
    parser.add_argument(
        "csv_path", type=str, help="Path to the telemetry CSV or dataset labels CSV."
    )
    args = parser.parse_args()

    plot_coverage(args.csv_path)
