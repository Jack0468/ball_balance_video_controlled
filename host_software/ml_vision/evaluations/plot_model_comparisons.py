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
        '95th_Percentile_Error_mm'
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
            
            if 'trial_' in model_name:
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
    
    # Create a large figure to hold subplots with plenty of white space
    fig, axes = plt.subplots(3, 2, figsize=(18, 16), facecolor='white')
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
