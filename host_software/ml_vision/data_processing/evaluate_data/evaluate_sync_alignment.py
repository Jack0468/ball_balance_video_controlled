import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="../../data/02_silver/session_20260730_174916/cnn_sequential_features.csv")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.abspath(os.path.join(script_dir, args.csv))

    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    print("Loading data...")
    df = pd.read_csv(csv_path)

    # Pick a 150 frame (5 second) continuous slice in the middle of the dataset
    start_idx = len(df) // 2
    slice_df = df.iloc[start_idx : start_idx + 150]
    
    # Normalize timestamps to start at 0
    t = slice_df['frame_timestamp_ms'] - slice_df['frame_timestamp_ms'].iloc[0]

    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(t, slice_df['touch_x'], label='True Telemetry (touch_x)', color='blue', linewidth=2)
    plt.plot(t, slice_df['cnn_x'], label='Vision Tracker (cnn_x)', color='orange', linestyle='--', linewidth=2)
    plt.title("X-Axis Synchronization Verification")
    plt.ylabel("Position (mm)")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(t, slice_df['touch_y'], label='True Telemetry (touch_y)', color='green', linewidth=2)
    plt.plot(t, slice_df['cnn_y'], label='Vision Tracker (cnn_y)', color='red', linestyle='--', linewidth=2)
    plt.title("Y-Axis Synchronization Verification")
    plt.xlabel("Time (ms)")
    plt.ylabel("Position (mm)")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    out_img = os.path.abspath(os.path.join(script_dir, "sync_verification.png"))
    plt.savefig(out_img, dpi=150)
    print(f"Saved sync verification plot to {out_img}")

if __name__ == '__main__':
    main()
