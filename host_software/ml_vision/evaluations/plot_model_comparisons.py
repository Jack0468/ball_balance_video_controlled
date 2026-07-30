import os
import json
import subprocess
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np

def run_benchmark_script(script_dir: str) -> None:
    benchmark_script = os.path.join(script_dir, 'benchmark_model_inference_times.py')
    if not os.path.exists(benchmark_script):
        raise FileNotFoundError(f"Benchmark script not found: {benchmark_script}")

    cmd = [sys.executable, benchmark_script, '--update_metrics', '--sample_fraction', '0.1', '--max_samples', '200']
    print(f"Running benchmark script: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=script_dir, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"Benchmark script failed with exit code {result.returncode}")


def main():
    parser = argparse.ArgumentParser(description='Plot model comparisons with optional inference benchmark')
    parser.add_argument('--benchmark', action='store_true', help='Run inference benchmark before plotting')
    parser.add_argument('--filter', type=str, choices=['yolo', 'resnet', 'mlp', 'all'], default='all', help='Filter which models to plot: yolo, resnet, mlp, or all')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.abspath(os.path.join(script_dir, '../models'))

    if args.benchmark:
        run_benchmark_script(script_dir)
    
    # Metrics we want to plot
    metric_keys = [
        'Mean_Euclidean_Error_mm',
        'RMSE_X_mm',
        'RMSE_Y_mm',
        'Max_Euclidean_Error_mm',
        '95th_Percentile_Error_mm',
        'Mean_Inference_Time_ms'
    ]
    
    model_names = []
    model_metrics = {key: [] for key in metric_keys}
    
    print(f"Searching for evaluation metrics in {models_dir}...")
    
    # Only consider top-level YOLO, ResNet, and MLP model directories and ignore archived contents.
    for model_name in sorted(os.listdir(models_dir)):
        model_root = os.path.join(models_dir, model_name)
        if not os.path.isdir(model_root):
            continue
        model_name_lower = model_name.lower()
        if model_name_lower == 'archive' or 'temporal' in model_name_lower:
            continue
            
        if args.filter != 'all':
            if args.filter not in model_name_lower:
                continue
        else:
            if ('yolo' not in model_name_lower and
                    'resnet' not in model_name_lower and
                    'mlp' not in model_name_lower and
                    'cnn' not in model_name_lower):
                continue

        json_path = None
        json_file = None
        for root, dirs, files in os.walk(model_root):
            if 'evaluation_metrics.json' in files or 'quick_evaluation_metrics.json' in files:
                json_file = 'evaluation_metrics.json' if 'evaluation_metrics.json' in files else 'quick_evaluation_metrics.json'
                json_path = os.path.join(root, json_file)
                break

        if json_path is None:
            print(f"No evaluation metrics found for {model_name}. Skipping.")
            continue

        with open(json_path, 'r') as f:
            try:
                data = json.load(f)

                model_names.append(model_name)
                for key in metric_keys:
                    # If a metric is missing (e.g. FPS), default to 0
                    val = data.get(key, 0.0)
                    model_metrics[key].append(val)

                print(f"Loaded metrics for {model_name} from {json_file}")
            except Exception as e:
                print(f"Error reading {json_path}: {e}")
                    
    if not model_names:
        print("No models with valid metrics found!")
        return
        
    print(f"\nPlotting comparisons for {len(model_names)} models...")
    
    # Adjust the subplot grid if inference time is included
    num_plots = len(metric_keys)
    rows = (num_plots + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(18, 8 * rows), facecolor='white')
    fig.suptitle('Model Evaluation Comparisons', fontsize=26, fontweight='bold', y=0.98, color='#424242', fontname='Arial')
    
    axes = axes.flatten()
    
    # Sydney Uni Monochromatic Red/Grey Theme
    sydney_colors = ['#E64626', '#FF7F50', '#808080', '#A9A9A9', '#C0C0C0', '#D3D3D3', '#E64626', '#8B0000', '#B22222', '#CD5C5C', '#696969']
    
    unique_models = sorted(list(set(model_names)))
    model_colors = {model: sydney_colors[j % len(sydney_colors)] for j, model in enumerate(unique_models)}
    
    for i, ax in enumerate(axes):
        if i >= len(metric_keys):
            ax.set_visible(False)
            continue
            
        metric = metric_keys[i]
        
        # Sort data for better visualization
        sorted_pairs = sorted(zip(model_names, model_metrics[metric]), key=lambda x: x[1])
        sorted_names, sorted_vals = zip(*sorted_pairs)
        
        # Assign consistent colors based on model name
        colors = [model_colors[name] for name in sorted_names]
        
        bars = ax.barh(sorted_names, sorted_vals, color=colors, edgecolor='none')
        
        # Add values to the end of bars
        for bar in bars:
            width = bar.get_width()
            ax.text(width + (max(sorted_vals) * 0.02), bar.get_y() + bar.get_height()/2, 
                    f'{width:.2f}', ha='left', va='center', fontsize=12, fontweight='bold', color='#424242', fontname='Arial')
                    
        title = metric.replace('_', ' ')
        ax.set_title(title, fontsize=18, fontweight='bold', color='#E64626', fontname='Arial', pad=15)
        if 'FPS' in metric:
            ax.set_xlabel('Frames Per Second', fontsize=14, fontname='Arial', color='#424242')
        elif 'Inference_Time' in metric:
            ax.set_xlabel('Inference Time (ms)', fontsize=14, fontname='Arial', color='#424242')
        else:
            ax.set_xlabel('Error in mm', fontsize=14, fontname='Arial', color='#424242')
            
        ax.grid(axis='x', linestyle='-', alpha=0.3, color='#808080')
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#808080')
        ax.spines['bottom'].set_color('#808080')
        ax.tick_params(axis='y', labelsize=12, colors='#424242')
        ax.tick_params(axis='x', labelsize=12, colors='#424242')
        
        # Filter out bbox models visually if they exist by fading them out
        for label in ax.get_yticklabels():
            if 'bbox' in label.get_text():
                label.set_color('#C0C0C0')
        ax.set_xlim(0, max(sorted_vals) * 1.15) # Add 15% padding for text
        ax.tick_params(axis='y', labelsize=12)
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the plot
    save_path = os.path.join(script_dir, 'model_comparisons.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved comparison graphs to {save_path}")
    
    try:
        plt.show()
    except:
        pass

if __name__ == '__main__':
    main()
