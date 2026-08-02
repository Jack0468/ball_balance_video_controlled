import os
import json
import subprocess
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np


def run_benchmark_script(script_dir: str) -> None:
    benchmark_script = os.path.join(script_dir, "benchmark_model_inference_times.py")
    if not os.path.exists(benchmark_script):
        raise FileNotFoundError(f"Benchmark script not found: {benchmark_script}")

    cmd = [
        sys.executable,
        benchmark_script,
        "--update_metrics",
        "--sample_fraction",
        "0.1",
        "--max_samples",
        "200",
    ]
    print(f"Running benchmark script: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=script_dir, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(
            f"Benchmark script failed with exit code {result.returncode}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Plot model comparisons with optional inference benchmark"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run inference benchmark before plotting",
    )
    parser.add_argument(
        "--filter",
        type=str,
        choices=["yolo", "resnet", "mlp", "cnn", "2d_tracker", "all_models", "all_metrics"],
        default="all_metrics",
        help="Filter which models to plot",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.abspath(os.path.join(script_dir, "../models"))

    if args.benchmark:
        run_benchmark_script(script_dir)

    if args.filter == "all_metrics":
        choices = ["yolo", "resnet", "mlp", "cnn", "2d_tracker", "all_models"]
        for choice in choices:
            print(f"\n--- Running plot for filter: {choice} ---")
            cmd = [sys.executable, os.path.abspath(__file__), "--filter", choice]
            subprocess.run(cmd)
        print("\nFinished generating all metrics plots.")
        return

    # Metrics we want to plot
    metric_keys = [
        "Mean_Euclidean_Error_mm",
        "RMSE_X_mm",
        "RMSE_Y_mm",
        "Max_Euclidean_Error_mm",
        "95th_Percentile_Error_mm",
        "Mean_Inference_Time_ms",
    ]

    model_names = []
    # We'll store full data dictionaries for each model
    model_data = {}

    print(f"Searching for evaluation metrics in {models_dir}...")

    # Load from PyTorch model directories
    for model_name in sorted(os.listdir(models_dir)):
        model_root = os.path.join(models_dir, model_name)
        if not os.path.isdir(model_root):
            continue
        model_name_lower = model_name.lower()
        if model_name_lower == "archive" or "temporal" in model_name_lower:
            continue

        if args.filter != "all_models":
            if args.filter not in model_name_lower:
                continue
        else:
            if (
                "yolo" not in model_name_lower
                and "resnet" not in model_name_lower
                and "mlp" not in model_name_lower
                and "cnn" not in model_name_lower
            ):
                continue

        json_path = None
        json_file = None
        for root, dirs, files in os.walk(model_root):
            if (
                "evaluation_metrics.json" in files
                or "quick_evaluation_metrics.json" in files
            ):
                json_file = (
                    "evaluation_metrics.json"
                    if "evaluation_metrics.json" in files
                    else "quick_evaluation_metrics.json"
                )
                json_path = os.path.join(root, json_file)
                break

        if json_path is None:
            continue

        with open(json_path, "r") as f:
            try:
                data = json.load(f)
                model_names.append(model_name)
                model_data[model_name] = data
                print(f"Loaded metrics for {model_name} from {json_file}")
            except Exception as e:
                print(f"Error reading {json_path}: {e}")
                
    # Additionally, check inference_time_summary.json for ONNX metrics or overriding metrics
    summary_json = os.path.join(script_dir, "inference_time_summary.json")
    if os.path.exists(summary_json):
        with open(summary_json, "r") as f:
            try:
                summary_data = json.load(f)
                for model in summary_data.get("models", []):
                    m_name = model["model_name"]
                    if m_name not in model_data:
                        model_names.append(m_name)
                        model_data[m_name] = model
                    else:
                        model_data[m_name].update(model)
            except Exception as e:
                print(f"Error reading {summary_json}: {e}")

    # Copy accuracy metrics from PyTorch models to ONNX models
    for m_name in list(model_data.keys()):
        if m_name.endswith("_ONNX"):
            base_name = m_name[:-5]
            if base_name in model_data:
                for metric in metric_keys:
                    if metric not in model_data[m_name] and metric in model_data[base_name]:
                        model_data[m_name][metric] = model_data[base_name][metric]

    if args.filter == "best_pipelines":
        model_names = [
            "ResNet (PyTorch)",
            "ResNet (ONNX)",
            "YOLO Pose (PyTorch)",
            "YOLO Pose (ONNX)",
            "YOLO + MLP",
            "Aruco + CNN",
            "Aruco + CNN + MLP",
            "YOLO + CNN",
            "YOLO + CNN + MLP"
        ]
        
        # Pull best models
        resnet_pt = model_data.get("resnet18_expert_tracker_0728_v6", {})
        resnet_onnx = model_data.get("resnet18_expert_tracker_0728_v6_ONNX", resnet_pt)
        yolo_pt = model_data.get("yolov8_platform_pose_markers_0728_v4", {})
        yolo_onnx = model_data.get("yolov8_platform_pose_markers_0728_v4_ONNX", yolo_pt)
        cnn = model_data.get("cnn_2d_tracker_0730_v3", {})
        mlp = model_data.get("mlp_corrector_0728_v6", {})

        aruco_time = 3.5 # Estimated ms for cv2.aruco.detectMarkers + Homography

        # Synthesize pipelines
        model_data = {}
        # 1. ResNet (Standalone)
        model_data["ResNet (PyTorch)"] = resnet_pt.copy()
        
        # 2. ResNet (ONNX)
        model_data["ResNet (ONNX)"] = resnet_onnx.copy()

        # 3. YOLO (Standalone)
        model_data["YOLO Pose (PyTorch)"] = yolo_pt.copy()

        # 4. YOLO (ONNX)
        model_data["YOLO Pose (ONNX)"] = yolo_onnx.copy()

        # 5. YOLO + MLP
        model_data["YOLO + MLP"] = yolo_pt.copy()
        model_data["YOLO + MLP"]["Mean_Inference_Time_ms"] = yolo_pt.get("Mean_Inference_Time_ms", 0) + mlp.get("Mean_Network_Time_ms", 1.0)
        model_data["YOLO + MLP"]["Mean_Postprocess_Time_ms"] = yolo_pt.get("Mean_Postprocess_Time_ms", 0) + mlp.get("Mean_Network_Time_ms", 1.0)
        model_data["YOLO + MLP"]["Mean_Euclidean_Error_mm"] = mlp.get("Mean_Euclidean_Error_mm", yolo_pt.get("Mean_Euclidean_Error_mm", 0))

        # 6. Aruco + CNN
        model_data["Aruco + CNN"] = cnn.copy()
        model_data["Aruco + CNN"]["Mean_Inference_Time_ms"] = cnn.get("Mean_Inference_Time_ms", 0) + aruco_time
        model_data["Aruco + CNN"]["Mean_Preprocess_Time_ms"] = cnn.get("Mean_Preprocess_Time_ms", 0) + aruco_time
        
        # 7. Aruco + CNN + MLP
        model_data["Aruco + CNN + MLP"] = model_data["Aruco + CNN"].copy()
        model_data["Aruco + CNN + MLP"]["Mean_Inference_Time_ms"] += mlp.get("Mean_Network_Time_ms", 1.0)
        model_data["Aruco + CNN + MLP"]["Mean_Postprocess_Time_ms"] = cnn.get("Mean_Postprocess_Time_ms", 0) + mlp.get("Mean_Network_Time_ms", 1.0)
        model_data["Aruco + CNN + MLP"]["Mean_Euclidean_Error_mm"] = mlp.get("Mean_Euclidean_Error_mm", cnn.get("Mean_Euclidean_Error_mm", 0))

        # 8. YOLO + CNN
        model_data["YOLO + CNN"] = cnn.copy()
        model_data["YOLO + CNN"]["Mean_Inference_Time_ms"] = cnn.get("Mean_Inference_Time_ms", 0) + yolo_pt.get("Mean_Inference_Time_ms", 0)
        model_data["YOLO + CNN"]["Mean_Preprocess_Time_ms"] = cnn.get("Mean_Preprocess_Time_ms", 0) + yolo_pt.get("Mean_Inference_Time_ms", 0)

        # 9. YOLO + CNN + MLP
        model_data["YOLO + CNN + MLP"] = model_data["YOLO + CNN"].copy()
        model_data["YOLO + CNN + MLP"]["Mean_Inference_Time_ms"] += mlp.get("Mean_Network_Time_ms", 1.0)
        model_data["YOLO + CNN + MLP"]["Mean_Postprocess_Time_ms"] = cnn.get("Mean_Postprocess_Time_ms", 0) + mlp.get("Mean_Network_Time_ms", 1.0)
        model_data["YOLO + CNN + MLP"]["Mean_Euclidean_Error_mm"] = mlp.get("Mean_Euclidean_Error_mm", cnn.get("Mean_Euclidean_Error_mm", 0))

    if not model_names:
        print("No models with valid metrics found!")
        return

    print(f"\nPlotting comparisons for {len(model_names)} models...")

    # Set up plots
    num_plots = len(metric_keys)
    rows = (num_plots + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(18, 8 * rows), facecolor="white")
    fig.suptitle(
        "Model Evaluation Comparisons",
        fontsize=26,
        fontweight="bold",
        y=0.98,
        color="#424242",
        fontname="Arial",
    )

    axes = axes.flatten()

    # Sydney Uni Monochromatic Red/Grey Theme
    sydney_colors = [
        "#E64626",
        "#FF7F50",
        "#808080",
        "#A9A9A9",
        "#C0C0C0",
        "#D3D3D3",
        "#E64626",
        "#8B0000",
        "#B22222",
        "#CD5C5C",
        "#696969",
    ]

    unique_models = sorted(list(set(model_names)))
    model_colors = {
        model: sydney_colors[j % len(sydney_colors)]
        for j, model in enumerate(unique_models)
    }

    for i, ax in enumerate(axes):
        if i >= len(metric_keys):
            ax.set_visible(False)
            continue

        metric = metric_keys[i]
        
        # Sort data for this metric
        vals = [model_data[name].get(metric, 0.0) for name in model_names]
        sorted_pairs = sorted(zip(model_names, vals), key=lambda x: x[1])
        sorted_names, sorted_vals = zip(*sorted_pairs)

        # Draw stacked bar chart for inference time, otherwise regular bar chart
        if metric == "Mean_Inference_Time_ms":
            # Stack components
            comp_colors = ["#424242", "#808080", "#E64626", "#FF7F50"]
            comp_labels = ["Disk IO", "Preprocess", "Network", "Postprocess"]
            comp_keys = ["Mean_Disk_IO_Time_ms", "Mean_Preprocess_Time_ms", "Mean_Network_Time_ms", "Mean_Postprocess_Time_ms"]
            
            y_pos = np.arange(len(sorted_names))
            left = np.zeros(len(sorted_names))
            
            bars = []
            for j, ckey in enumerate(comp_keys):
                cvals = [model_data[name].get(ckey, 0.0) for name in sorted_names]
                b = ax.barh(y_pos, cvals, left=left, color=comp_colors[j], edgecolor="none", label=comp_labels[j])
                left += cvals
                bars.append(b)
                
            ax.set_yticks(y_pos)
            ax.set_yticklabels(sorted_names)
            ax.legend(loc="lower right")
            
            # Label total sum at the end
            for y, val in zip(y_pos, sorted_vals):
                if val > 0:
                    ax.text(val + (max(sorted_vals) * 0.02), y, f"{val:.2f}", ha="left", va="center", fontsize=12, fontweight="bold", color="#424242", fontname="Arial")
        else:
            colors = [model_colors[name] for name in sorted_names]
            bars = ax.barh(sorted_names, sorted_vals, color=colors, edgecolor="none")
            
            for bar in bars:
                width = bar.get_width()
                if width > 0:
                    ax.text(
                        width + (max(sorted_vals) * 0.02),
                        bar.get_y() + bar.get_height() / 2,
                        f"{width:.2f}",
                        ha="left",
                        va="center",
                        fontsize=12,
                        fontweight="bold",
                        color="#424242",
                        fontname="Arial",
                    )

        title = metric.replace("_", " ")
        ax.set_title(
            title,
            fontsize=18,
            fontweight="bold",
            color="#E64626",
            fontname="Arial",
            pad=15,
        )
        if "FPS" in metric:
            ax.set_xlabel("Frames Per Second", fontsize=14, fontname="Arial", color="#424242")
        elif "Time_ms" in metric:
            ax.set_xlabel("Inference Time (ms)", fontsize=14, fontname="Arial", color="#424242")
        else:
            ax.set_xlabel("Error in mm", fontsize=14, fontname="Arial", color="#424242")

        ax.grid(axis="x", linestyle="-", alpha=0.3, color="#808080")
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#808080")
        ax.spines["bottom"].set_color("#808080")
        ax.tick_params(axis="y", labelsize=12, colors="#424242")
        ax.tick_params(axis="x", labelsize=12, colors="#424242")

        for label in ax.get_yticklabels():
            if "bbox" in label.get_text():
                label.set_color("#C0C0C0")
        if max(sorted_vals) > 0:
            ax.set_xlim(0, max(sorted_vals) * 1.15) 
        ax.tick_params(axis="y", labelsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(script_dir, f"model_comparisons_{args.filter}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved comparison graphs to {save_path}")

    pass


if __name__ == "__main__":
    main()
