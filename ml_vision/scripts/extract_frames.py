"""
extract_frames.py

A simple utility script to convert an MP4 video into individual PNG frames 
for preprocessing tests when CSV label data is not available.
"""

import cv2
import os
import argparse

def extract_frames(video_path, output_dir, max_frames=None):
    """
    Reads a video and saves its frames as PNG images.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video loaded: {frame_count} frames at {fps} FPS")

    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Stop if we reach max_frames limit
        if max_frames is not None and saved_count >= max_frames:
            break

        # Save frame as PNG
        frame_filename = os.path.join(output_dir, f"frame_{frame_idx:05d}.png")
        cv2.imwrite(frame_filename, frame)

        saved_count += 1
        frame_idx += 1
        
        if saved_count % 100 == 0:
            print(f"Extracted {saved_count} frames...")

    cap.release()
    print(f"Extraction complete. Saved {saved_count} PNG frames to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from a video file.")
    parser.add_argument("video_path", help="Path to the input video file (.mp4)")
    parser.add_argument("output_dir", help="Directory to save the extracted PNG frames")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum number of frames to extract")
    
    args = parser.parse_args()
    
    extract_frames(args.video_path, args.output_dir, args.max_frames)
