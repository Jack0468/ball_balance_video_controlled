import cv2
import pandas as pd
import argparse
import sys
import os
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Play through a cleaned dataset to visually verify telemetry.")
    parser.add_argument('--data_dir', required=True, help="Path to dataset directory (e.g. ../data/02_silver)")
    parser.add_argument('--csv_name', default='labels_sequential.csv', help="Name of the telemetry CSV file")
    parser.add_argument('--output', default='dataset_playback.mp4', help="Output mp4 path (default: dataset_playback.mp4)")
    parser.add_argument('--max_frames', type=int, default=1500, help="Max frames to play (default: 1500, roughly 50s)")
    args = parser.parse_args()

    csv_path = os.path.join(args.data_dir, args.csv_name)
    images_dir = os.path.join(args.data_dir, 'images')

    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}")
        sys.exit(1)

    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    if len(df) == 0:
        print("Error: Dataset is empty.")
        sys.exit(1)
        
    print(f"Dataset has {len(df)} frames. Generating playback for up to {args.max_frames} frames...")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, 30.0, (640, 480))
    
    frames_to_process = min(len(df), args.max_frames)
    
    for i in tqdm(range(frames_to_process)):
        row = df.iloc[i]
        
        img_path = os.path.join(images_dir, row['image_file'])
        if not os.path.exists(img_path):
            print(f"\nWarning: Image {img_path} not found. Skipping.")
            continue
            
        frame = cv2.imread(img_path)
        if frame is None:
            continue
            
        # Ensure it's 640x480 (the dataset should already be, but just in case)
        if frame.shape[:2] != (480, 640):
            frame = cv2.resize(frame, (640, 480))
            
        # Get telemetry touch coordinates
        tx = float(row.get('touch_x', 0))
        ty = float(row.get('touch_y', 0))
        
        # Map mm coordinates to 640x480 pixels, treating (0,0) as the exact center (320, 240)
        # Platform is 140mm wide (-70 to 70) and 110mm tall (-55 to 55)
        # Note: In OpenCV, Y=0 is the TOP of the image. 
        px = int(320 + (tx / 140.0) * 640)
        py = int(240 - (ty / 110.0) * 480)
        
        # Get telemetry target coordinates (the goal)
        bx = float(row.get('target_x', 0))
        by = float(row.get('target_y', 0))
        bpx = int(320 + (bx / 140.0) * 640)
        bpy = int(240 - (by / 110.0) * 480)
        
        # Draw Touch (Red)
        cv2.circle(frame, (px, py), 15, (0, 0, 255), 3)
        # Draw Ball Target (Green)
        cv2.circle(frame, (bpx, bpy), 15, (0, 255, 0), 3)
        
        # Annotations
        cv2.putText(frame, f"Touch X:{tx:.1f} Y:{ty:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.putText(frame, f"Target X:{bx:.1f} Y:{by:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, f"Dataset Row: {i} | Image: {row['image_file']}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        out.write(frame)

    out.release()
    print(f"Done! Saved verification video to {args.output}")

if __name__ == "__main__":
    main()
