import pandas as pd
import argparse
import os

def clean_dataset(data_dir):
    csv_path = os.path.join(data_dir, 'labels.csv')
    out_path = os.path.join(data_dir, 'labels_sequential.csv')
    
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}")
        return
        
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    initial_count = len(df)
    print(f"Initial rows: {initial_count}")
    
    # Drop duplicates based on 'image_file', keeping the first occurrence
    # This guarantees exactly 1 row per video frame, maintaining chronological order
    df_cleaned = df.drop_duplicates(subset=['image_file'], keep='first').copy()
    
    unique_count = len(df_cleaned)
    print(f"Removed {initial_count - unique_count} duplicate image rows.")
    
    # Identify and drop "frozen" telemetry frames (where the 1.5s debouncer kicked in because the ball fell off/bounced)
    # Since real physical balls always have ADC noise > 0.1mm, perfectly identical coordinates mean the ball is missing.
    if 'touch_x' in df_cleaned.columns and 'touch_y' in df_cleaned.columns:
        # Check if current row is identical to previous row
        is_frozen = (df_cleaned['touch_x'] == df_cleaned['touch_x'].shift(1)) & (df_cleaned['touch_y'] == df_cleaned['touch_y'].shift(1))
        
        frozen_count = is_frozen.sum()
        df_cleaned = df_cleaned[~is_frozen]
        print(f"Removed {frozen_count} frozen telemetry frames (ball off-board or bouncing).")
        
    final_count = len(df_cleaned)
    print(f"Final Cleaned rows: {final_count}")
    
    df_cleaned.to_csv(out_path, index=False)
    print(f"Saved cleaned dataset to {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Deduplicate telemetry to 1 row per image")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to dataset directory (e.g. ../data/02_silver)")
    
    args = parser.parse_args()
    clean_dataset(args.data_dir)
