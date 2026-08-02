import os
import json
import matplotlib.pyplot as plt
import numpy as np

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.abspath(os.path.join(script_dir, "../models"))

    pipelines = [
        {
            "name": "Aruco (Pure OpenCV)",
            "model_dir": None,
            "latency_breakdown": {"Aruco": 2.0},
            "metrics": {
                "Mean_Euclidean_Error_mm": 113.2,
                "Mean_Inference_Time_ms": 2.0
            }
        },
        {
            "name": "YOLO + MLP (iPhone Baseline, PyTorch)",
            "model_dir": "mlp_corrector_iphone_v1",
            "latency_breakdown": {"YOLO": 14.2, "MLP": None}
        },
        {
            "name": "YOLO + CNN + MLP (Fixed Angle, PyTorch)",
            "model_dir": "mlp_corrector_0728_v2",
            "latency_breakdown": {"YOLO": 15.1, "CNN Tracker": 8.5, "MLP": None}
        },
        {
            "name": "Aruco + MLP (Synthetic, PyTorch)",
            "model_dir": "mlp_corrector_time_0730_v1",
            "latency_breakdown": {"Aruco": 2.0, "MLP": None}
        },
        {
            "name": "Aruco + CNN + MLP (Best System, ONNX)",
            "model_dir": "mlp_corrector_time_aruco_0730_v1",
            "latency_breakdown": {"Aruco": 2.0, "CNN Tracker": 23.6, "MLP": 0.1}
        },
        {
            "name": "ResNet Expert (End-to-End, PyTorch)",
            "model_dir": "resnet18_expert_tracker_0730_v7",
            "latency_breakdown": {"ResNet": None}
        }
    ]

    plot_data = []

    for p in pipelines:
        if p["model_dir"] is not None:
            metrics_path = os.path.join(models_dir, p["model_dir"], "evaluation_metrics.json")
            if not os.path.exists(metrics_path):
                print(f"Warning: {metrics_path} not found. Skipping {p['name']}")
                continue
            with open(metrics_path, "r") as f:
                data = json.load(f)
            
            error = data.get("Mean_Euclidean_Error_mm")
            
            # Prefer Mean_Network_Time_ms for the raw model latency, fallback to Inference_Time
            mlp_time = data.get("Mean_Network_Time_ms", data.get("Mean_Inference_Time_ms"))
            
            if error is None or mlp_time is None:
                continue

            # Fill in the latency breakdown dynamically for the model component
            for k, v in p["latency_breakdown"].items():
                if v is None:
                    p["latency_breakdown"][k] = mlp_time

            total_latency = sum(p["latency_breakdown"].values())
            
            plot_data.append({
                "name": p["name"],
                "error": error,
                "latency": total_latency,
                "breakdown": p["latency_breakdown"]
            })
        else:
            plot_data.append({
                "name": p["name"],
                "error": p["metrics"]["Mean_Euclidean_Error_mm"],
                "latency": sum(p["latency_breakdown"].values()),
                "breakdown": p["latency_breakdown"]
            })

    if not plot_data:
        print("No pipeline data found to plot.")
        return

    # Sort by error (best to worst)
    plot_data.sort(key=lambda x: x["error"])

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Pipeline Performance Dashboard", fontsize=16, fontweight='bold')

    # --- Subplot 1: Pareto Frontier (Accuracy vs Speed) ---
    for item in plot_data:
        ax1.scatter(item["latency"], item["error"], s=100, label=item["name"])
        # Annotate
        ax1.annotate(item["name"], (item["latency"], item["error"]), 
                     xytext=(5, 5), textcoords='offset points', fontsize=9)

    ax1.set_xlabel("Mean Inference Time (ms) -> LOWER is better", fontsize=12)
    ax1.set_ylabel("Mean Euclidean Error (mm) -> LOWER is better", fontsize=12)
    ax1.set_title("Speed vs. Accuracy Tradeoff", fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Invert x and y axis if we want "up and right" to be better, 
    # but traditionally, bottom-left is the pareto front.
    # Let's highlight the pareto front.
    sorted_by_latency = sorted(plot_data, key=lambda x: x["latency"])
    front_x = []
    front_y = []
    min_error = float('inf')
    for item in sorted_by_latency:
        if item["error"] < min_error:
            front_x.append(item["latency"])
            front_y.append(item["error"])
            min_error = item["error"]
    
    ax1.plot(front_x, front_y, '--', color='gray', alpha=0.5, label='Pareto Frontier')
    ax1.legend()

    # --- Subplot 2: Latency Breakdown ---
    names = [item["name"] for item in plot_data]
    y_pos = np.arange(len(names))
    
    # Collect all unique components
    components = []
    for item in plot_data:
        for k in item["breakdown"].keys():
            if k not in components:
                components.append(k)
    
    # Bottom tracker for stacked bar
    bottoms = np.zeros(len(plot_data))
    
    colors = plt.get_cmap('tab10')(np.linspace(0, 1, len(components)))
    
    for i, component in enumerate(components):
        values = [item["breakdown"].get(component, 0) for item in plot_data]
        ax2.barh(y_pos, values, left=bottoms, color=colors[i], label=component, edgecolor='white')
        bottoms += np.array(values)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(names)
    ax2.invert_yaxis()  # top-to-bottom
    ax2.set_xlabel("Latency (ms)", fontsize=12)
    ax2.set_title("Inference Latency Breakdown", fontsize=14)
    ax2.legend(title="Pipeline Stages")
    
    # Add total latency annotations
    for i, item in enumerate(plot_data):
        ax2.text(item["latency"] + 0.2, i, f'{item["latency"]:.1f}ms', va='center', fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    output_path = os.path.join(script_dir, "pipeline_performance_dashboard.png")
    plt.savefig(output_path, dpi=300)
    print(f"Saved dashboard to {output_path}")

if __name__ == "__main__":
    main()
