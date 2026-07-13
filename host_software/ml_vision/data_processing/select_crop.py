import cv2
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Helper script to select a cropping ROI from a video.")
    parser.add_argument('--video', required=True, help="Path to the video file")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video {args.video}")
        sys.exit(1)

    # Read a few frames in to get past any black/blank starting frames
    for _ in range(30):
        ret, frame = cap.read()
        if not ret:
            break
            
    if not ret or frame is None:
        print("Error: Could not read frame from video.")
        sys.exit(1)

    # We resize it slightly if it's too huge for the screen, but then we must scale the crop back.
    # Actually, cv2.selectROI handles window sizing nicely if we just pass the frame.
    print("\n--- ROI Selection ---")
    print("1. Click and drag to draw a bounding box around the platform.")
    print("2. Press SPACE or ENTER to confirm your selection.")
    print("3. Press 'c' to cancel and try again.")
    print("---------------------\n")
    
    # Scale the image down so it easily fits on a laptop screen
    scale = 0.4
    display_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
    
    # False for showCrosshair, False for fromCenter
    roi = cv2.selectROI("Select Crop Region", display_frame, False, False)
    cv2.destroyAllWindows()

    if roi == (0, 0, 0, 0):
        print("Selection cancelled or empty.")
    else:
        # Scale coordinates back to original video resolution
        x, y, w, h = [int(v / scale) for v in roi]
        print("\n=============================================")
        print(f"Selected Crop Box (x, y, w, h): {x},{y},{w},{h}")
        print("=============================================\n")
        print("You can now run the sync script with this crop parameter:")
        print(f"python host_software/ml_vision/scripts/sync_telemetry_video.py --video {args.video} --telemetry <path> --output <dir> --crop {x},{y},{w},{h}")

if __name__ == "__main__":
    main()
