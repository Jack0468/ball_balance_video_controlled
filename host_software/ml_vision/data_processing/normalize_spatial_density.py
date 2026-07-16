import pandas as pd
import numpy as np
import argparse
import os

def normalize_dataset(csv_path, max_samples=50):
    print(f"Reading dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    
    x_col, y_col = None, None
    if 'touch_x' in df.columns and 'touch_y' in df.columns:
        x_col, y_col = 'touch_x', 'touch_y'
    elif 'target_x' in df.columns and 'target_y' in df.columns:
        x_col, y_col = 'target_x', 'target_y'
    else:
        print("Error: Could not find spatial columns.")
        return

    safe_x_range = [-80.0, 80.0]
    safe_y_range = [-60.0, 60.0]
    grid_size_mm = 10.0
    
    # Calculate grid indices for each row
    df['grid_x'] = np.floor((df[x_col] - safe_x_range[0]) / grid_size_mm)
    df['grid_y'] = np.floor((df[y_col] - safe_y_range[0]) / grid_size_mm)
    
    # Filter out rows outside the safe zone?
    # We might want to keep everything outside the safe zone as they are rare.
    # We will only subsample rows inside the safe zone.
    inside_mask = (
        (df[x_col] >= safe_x_range[0]) & (df[x_col] <= safe_x_range[1]) &
        (df[y_col] >= safe_y_range[0]) & (df[y_col] <= safe_y_range[1])
    )
    
    df_inside = df[inside_mask]
    df_outside = df[~inside_mask]
    
    print(f"Total rows: {len(df)}")
    print(f"Rows inside safe zone: {len(df_inside)}")
    print(f"Rows outside safe zone: {len(df_outside)}")
    
    # Group by grid cells and sample up to max_samples
    # We use random state to ensure reproducibility
    sampled_inside = df_inside.groupby(['grid_x', 'grid_y'], group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), max_samples), random_state=42)
    )
    
    print(f"Sampled rows inside safe zone: {len(sampled_inside)} (dropped {len(df_inside) - len(sampled_inside)})")
    
    # Combine back with outside rows
    df_normalized = pd.concat([sampled_inside, df_outside])
    
    # Re-sort chronologically to maintain temporal split consistency
    if 'host_timestamp_ms' in df_normalized.columns:
        df_normalized = df_normalized.sort_values('host_timestamp_ms')
    
    # Drop temp grid columns
    df_normalized = df_normalized.drop(columns=['grid_x', 'grid_y'])
    
    out_path = os.path.join(os.path.dirname(csv_path), 'labels_normalized.csv')
    df_normalized.to_csv(out_path, index=False)
    
    print(f"Saved normalized dataset to: {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Normalize dataset spatial density")
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--max-samples", type=int, default=50, help="Max samples per 10x10mm cell")
    args = parser.parse_args()
    
    normalize_dataset(args.csv_path, args.max_samples)
