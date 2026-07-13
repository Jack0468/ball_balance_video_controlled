import cv2
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Interactive script to find the exact frame and timestamp.")
    parser.add_argument('--video', required=True, help="Path to the .MOV video file")
    args = parser.parse_args()

    print(f"Opening video: {args.video}")
    cap = cv2.VideoCapture(args.video)
    
    if not cap.isOpened():
        print(f"Error: Could not open {args.video}")
        sys.exit(1)
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0
    
    print("\n=======================================================")
    print("INTERACTIVE TIMESTAMP FINDER")
    print("Controls:")
    print("  [d] - Step Forward 1 Frame")
    print("  [a] - Step Backward 1 Frame")
    print("  [e] - Step Forward 30 Frames (1 Second)")
    print("  [q] - Step Backward 30 Frames (1 Second)")
    print("  [ENTER] or [ESC] - Quit")
    print("\nFind the exact frame where the giant green timestamp appears.")
    print("Write down the 'Frame Index' and the 'Green Timestamp'!")
    print("=======================================================\n")
    
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        
        if not ret:
            current_frame = total_frames - 1
            continue
            
        fh, fw = frame.shape[:2]
        
        # Scale down for display if the video is huge
        scale = 0.5
        display_frame = cv2.resize(frame, (int(fw * scale), int(fh * scale)))
        
        # Draw frame index on screen
        cv2.putText(display_frame, f"Frame Index: {current_frame}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                    
        cv2.imshow("Interactive Timestamp Finder", display_frame)
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('d'):
            current_frame = min(total_frames - 1, current_frame + 1)
        elif key == ord('a'):
            current_frame = max(0, current_frame - 1)
        elif key == ord('e'):
            current_frame = min(total_frames - 1, current_frame + 30)
        elif key == ord('q'):
            current_frame = max(0, current_frame - 30)
        elif key == 13 or key == 27: # Enter or ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nStopped at Frame Index: {current_frame}")
    print("Please use this frame index and the visible green timestamp in sync_data.py!")

if __name__ == "__main__":
    main()
