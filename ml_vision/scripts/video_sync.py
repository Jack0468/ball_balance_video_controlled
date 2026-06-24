"""
video_sync.py

This script reads MP4 video files and extracts frames. It also parses a CSV file
containing the (x, y) position of the ball and uses linear interpolation to 
synchronize the position data with the exact timestamps of the video frames.
"""

import cv2
import pandas as pd
import numpy as np
import os

def sync_data(video_path, csv_path, output_dir):
    """
    Reads video and CSV, synchronizes the data, and saves labeled frames.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Load CSV and prepare for interpolation
    # Assuming CSV has columns: ['timestamp', 'x', 'y']
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return

    # 2. Open Video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("Error: Could not determine video FPS.")
        return
        
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video loaded: {frame_count} frames at {fps} FPS")

    # 3. Process video and interpolate data
    synced_data = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate exact timestamp for this frame
        current_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        current_time_s = current_time_ms / 1000.0

        # Linear interpolation for (x, y)
        if not df.empty and 'timestamp' in df.columns:
            # np.interp requires x coordinates to be increasing
            x_interp = np.interp(current_time_s, df['timestamp'], df['x'])
            y_interp = np.interp(current_time_s, df['timestamp'], df['y'])
        else:
            x_interp, y_interp = None, None

        # Save frame
        frame_filename = os.path.join(output_dir, f"frame_{frame_idx:05d}.jpg")
        cv2.imwrite(frame_filename, frame)

        # Store synced data
        synced_data.append({
            'frame_idx': frame_idx,
            'timestamp': current_time_s,
            'frame_path': frame_filename,
            'x': x_interp,
            'y': y_interp
        })

        frame_idx += 1

    cap.release()

    # Save synced data to a new CSV
    synced_df = pd.DataFrame(synced_data)
    synced_csv_path = os.path.join(output_dir, "synced_labels.csv")
    synced_df.to_csv(synced_csv_path, index=False)
    print(f"Synchronization complete. Saved {len(synced_data)} frames and labels to {output_dir}")

if __name__ == "__main__":
    # Example usage
    # sync_data("data/video/sample.mp4", "data/labels.csv", "data/processed_frames")
    pass
