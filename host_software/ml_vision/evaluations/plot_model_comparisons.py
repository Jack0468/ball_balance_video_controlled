import os
import json
import matplotlib.pyplot as plt
import numpy as np

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.abspath(os.path.join(script_dir, '../models'))
    
    # Metrics we want to plot
    metric_keys = [
        'Mean_Euclidean_Error_mm',
        'RMSE_X_mm',
        'RMSE_Y_mm',
        'Max_Euclidean_Error_mm',
        '95th_Percentile_Error_mm',
        'FPS_Estimate'
    ]
    
    model_names = []
    model_metrics = {key: [] for key in metric_keys}
    
    print(f"Searching for evaluation metrics in {models_dir}...")
    
    # Traverse directories to find evaluation_metrics.json
    for root, dirs, files in os.walk(models_dir):
        if 'evaluation_metrics.json' in files or 'quick_evaluation_metrics.json' in files:
            json_file = 'evaluation_metrics.json' if 'evaluation_metrics.json' in files else 'quick_evaluation_metrics.json'
            json_path = os.path.join(root, json_file)
            model_name = os.path.basename(root)
            
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
    
    # Create a large figure to hold subplots
    fig, axes = plt.subplots(3, 2, figsize=(18, 16))
    fig.suptitle('Model Evaluation Comparisons', fontsize=22, fontweight='bold', y=0.98)
    
    axes = axes.flatten()
    
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(model_names)))
    
    for i, metric in enumerate(metric_keys):
        ax = axes[i]
        
        # Sort data for better visualization
        # For FPS, higher is better, so we can sort ascending to keep the best at the top or bottom
        sorted_pairs = sorted(zip(model_names, model_metrics[metric]), key=lambda x: x[1])
        sorted_names, sorted_vals = zip(*sorted_pairs)
        
        bars = ax.barh(sorted_names, sorted_vals, color=colors)
        
        # Add values to the end of bars
        for bar in bars:
            width = bar.get_width()
            ax.text(width + (width * 0.02), bar.get_y() + bar.get_height()/2, 
                    f'{width:.2f}', ha='left', va='center', fontsize=11, fontweight='bold')
                    
        title = metric.replace('_', ' ')
        ax.set_title(title, fontsize=16, fontweight='bold')
        if 'FPS' in metric:
            ax.set_xlabel('Frames Per Second (Higher is better)', fontsize=12)
        else:
            ax.set_xlabel('Error in mm (Lower is better)', fontsize=12)
            
        ax.grid(axis='x', linestyle='--', alpha=0.7)
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
