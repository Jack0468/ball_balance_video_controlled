import cv2
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Extract a 30s clip from bronze video.")
    parser.add_argument("--input", default="../data/01_bronze/video1/20260710_054604000_iOS.MOV")
    parser.add_argument("--output", default="../bronze_demo.mp4")
    parser.add_argument("--duration", type=int, default=30, help="Duration in seconds")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, args.input) if not os.path.isabs(args.input) else args.input
    output_path = os.path.join(script_dir, args.output) if not os.path.isabs(args.output) else args.output

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    target_frames = int(fps * args.duration)
    
    print(f"Input Video: {input_path}")
    print(f"Resolution: {width}x{height} @ {fps} FPS")
    print(f"Extracting {target_frames} frames ({args.duration} seconds)...")

    # Use mp4v codec for standard mp4 format
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frames_written = 0
    while frames_written < target_frames:
        ret, frame = cap.read()
        if not ret:
            print("Reached end of video before reaching target duration.")
            break
            
        out.write(frame)
        frames_written += 1
        
        if frames_written % int(fps) == 0:
            print(f"Processed {frames_written // fps} / {args.duration} seconds...")

    cap.release()
    out.release()
    print(f"\nDone! Saved {args.duration}-second clip to {output_path}")

if __name__ == "__main__":
    main()
