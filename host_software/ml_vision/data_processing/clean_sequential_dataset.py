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
    df_cleaned = df.drop_duplicates(subset=['image_file'], keep='first')
    
    final_count = len(df_cleaned)
    print(f"Cleaned rows (Unique frames): {final_count}")
    print(f"Removed {initial_count - final_count} duplicate telemetry rows.")
    
    df_cleaned.to_csv(out_path, index=False)
    print(f"Saved cleaned dataset to {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Deduplicate telemetry to 1 row per image")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to dataset directory (e.g. ../data/02_silver)")
    
    args = parser.parse_args()
    clean_dataset(args.data_dir)
