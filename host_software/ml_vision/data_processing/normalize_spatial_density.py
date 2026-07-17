import pandas as pd
import numpy as np
import argparse
import os

def normalize_dataset(csv_path):
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

    grid_size_mm = 5.0
    
    # Calculate grid indices for each row
    df['grid_x'] = np.floor(df[x_col] / grid_size_mm)
    df['grid_y'] = np.floor(df[y_col] / grid_size_mm)
    
    cell_counts = df.groupby(['grid_x', 'grid_y']).size()
    
    # Determine the outliers in terms of frequency
    Q1 = cell_counts.quantile(0.25)
    Q3 = cell_counts.quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    
    # The frequency of the majority of the dataset
    non_outlier_counts = cell_counts[cell_counts <= upper_bound]
    majority_freq = int(non_outlier_counts.median()) if not non_outlier_counts.empty else 1
    majority_freq = max(1, majority_freq) # Ensure at least 1
    
    print(f"Total rows: {len(df)}")
    print(f"Grid cell frequencies - min: {cell_counts.min()}, max: {cell_counts.max()}, median: {int(cell_counts.median())}")
    print(f"Frequency outlier threshold (Q3 + 1.5*IQR): {upper_bound:.1f}")
    print(f"Target majority frequency: {majority_freq}")
    
    # Group by grid cells and sample up to majority_freq
    df_normalized = df.groupby(['grid_x', 'grid_y'], group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), majority_freq), random_state=42),
        include_groups=False
    )
    
    print(f"Rows after normalization: {len(df_normalized)} (dropped {len(df) - len(df_normalized)})")
    
    # Re-sort chronologically to maintain temporal split consistency
    if 'host_timestamp_ms' in df_normalized.columns:
        df_normalized = df_normalized.sort_values('host_timestamp_ms')
    
    # Drop temp grid columns (if they are still present)
    df_normalized = df_normalized.drop(columns=['grid_x', 'grid_y'], errors='ignore')
    
    out_path = os.path.join(os.path.dirname(csv_path), 'labels_normalized.csv')
    df_normalized.to_csv(out_path, index=False)
    
    print(f"Saved normalized dataset to: {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Normalize dataset spatial density")
    parser.add_argument("--csv_path", required=True)
    args = parser.parse_args()
    
    normalize_dataset(args.csv_path)
