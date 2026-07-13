import cv2
import pandas as pd
import numpy as np
import argparse
import sys
import os

class DataSynchronizer:
    def __init__(self, video_path, telemetry_path, sync_frame, sync_timestamp):
        self.video_path = video_path
        self.telemetry_path = telemetry_path
        self.sync_frame = int(sync_frame)
        self.sync_timestamp = int(sync_timestamp)
        
    def run_sync(self, output_csv_path):
        print(f"Loading telemetry from {self.telemetry_path}...")
        df_tel = pd.read_csv(self.telemetry_path)
        
        # Ensure telemetry is sorted by host_timestamp_ms for fast binary search
        df_tel = df_tel.sort_values(by='host_timestamp_ms').reset_index(drop=True)
        
        print(f"Opening video {self.video_path}...")
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            sys.exit(1)
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Collect video frame timestamps
        print("Reading Variable Frame Rate (VFR) timestamps from video...")
        frame_timestamps = []
        sync_msec = 0
        for i in range(total_frames):
            ret = cap.grab()
            if not ret:
                break
            
            # Get the exact presentation timestamp in milliseconds
            pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            
            frame_timestamps.append({
                'frame_index': i,
                'pos_msec': pos_msec
            })
            
            if i == self.sync_frame:
                sync_msec = pos_msec
                
        cap.release()
        
        # Calculate absolute time for each frame
        for ft in frame_timestamps:
            relative_ms = ft['pos_msec'] - sync_msec
            ft['video_time_ms'] = self.sync_timestamp + relative_ms
        
        df_video = pd.DataFrame(frame_timestamps)
        
        print("Aligning telemetry to video frames...")
        # For each video frame, find the closest telemetry row
        tel_times = df_tel['host_timestamp_ms'].values
        video_times = df_video['video_time_ms'].values
        
        # Find insertion indices
        indices = np.searchsorted(tel_times, video_times, side='left')
        
        # Clamp indices
        indices = np.clip(indices, 0, len(tel_times) - 1)
        
        # For indices > 0, check if the previous element was actually closer
        left_closer = (indices > 0) & (
            np.abs(video_times - tel_times[indices - 1]) < np.abs(video_times - tel_times[indices])
        )
        indices[left_closer] -= 1
        
        # Build the perfectly aligned dataset
        df_aligned = df_tel.iloc[indices].copy().reset_index(drop=True)
        df_aligned.insert(0, 'frame_index', df_video['frame_index'])
        df_aligned.insert(1, 'video_time_ms', df_video['video_time_ms'])
        
        df_aligned.to_csv(output_csv_path, index=False)
        print(f"Successfully wrote synchronized mapping to {output_csv_path}")

def main():
    parser = argparse.ArgumentParser(description="Synchronize telemetry data with video VFR timestamps.")
    parser.add_argument('--video', required=True, help="Path to the raw .MOV video")
    parser.add_argument('--telemetry', required=True, help="Path to the raw telemetry .csv")
    parser.add_argument('--sync-frame', required=True, type=int, help="The exact video frame index where the timestamp was read")
    parser.add_argument('--sync-timestamp', required=True, type=int, help="The exact Unix timestamp read from the screen at sync-frame")
    parser.add_argument('--output', required=True, help="Path to output the synced CSV mapping")
    
    args = parser.parse_args()
    
    sync = DataSynchronizer(args.video, args.telemetry, args.sync_frame, args.sync_timestamp)
    sync.run_sync(args.output)

if __name__ == "__main__":
    main()
