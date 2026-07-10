import cv2
import pandas as pd
import numpy as np
import os
import argparse
from datetime import datetime, timezone

def auto_detect_crop(video_path, num_frames=100, padding=50):
    print(f"Auto-detecting platform bounding box over the first {num_frames} frames...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video for auto-crop.")
        return None
        
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = 0, 0
    valid_frames = 0
    
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        kernel = np.ones((5,5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
            
        # Find the large contour that is closest to the center of the image
        fh, fw = frame.shape[:2]
        img_cx, img_cy = fw / 2, fh / 2
        
        best_cnt = None
        min_dist = float('inf')
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 20000: # Ensure it's a reasonably large object
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    # Distance to center
                    dist = (cx - img_cx)**2 + (cy - img_cy)**2
                    if dist < min_dist:
                        min_dist = dist
                        best_cnt = cnt
                        
        if best_cnt is not None:
            x, y, w, h = cv2.boundingRect(best_cnt)
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x + w)
            max_y = max(max_y, y + h)
            valid_frames += 1
            
    cap.release()
    
    if valid_frames == 0:
        print("Warning: Could not auto-detect platform. Disabling crop.")
        return None
        
    # Read one frame to get max dimensions for padding limits
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    fh, fw = frame.shape[:2]
    
    cx = max(0, min_x - padding)
    cy = max(0, min_y - padding)
    cw = min(fw - cx, (max_x - min_x) + 2 * padding)
    ch = min(fh - cy, (max_y - min_y) + 2 * padding)
    
    crop_box = (cx, cy, cw, ch)
    print(f"Auto-crop box determined: {crop_box} (x, y, w, h)")
    
    # Show the user the bounding box to confirm
    preview = frame.copy()
    cv2.rectangle(preview, (cx, cy), (cx+cw, cy+ch), (0, 255, 0), 4)
    cv2.namedWindow("Verify Auto-Crop", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Verify Auto-Crop", fw // 2, fh // 2)
    cv2.imshow("Verify Auto-Crop", preview)
    
    print("\n--- Verify Auto-Crop ---")
    print("Press 'y' or ENTER to accept this bounding box.")
    print("Press 'n' to reject and cancel.")
    print("------------------------\n")
    
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('y') or key == 13 or key == 32: # y, ENTER, or SPACE
            print("Crop confirmed by user.")
            cv2.destroyAllWindows()
            break
        elif key == ord('n') or key == 27 or key == ord('c'): # n, ESC, or c
            print("Crop rejected by user. Exiting.")
            cv2.destroyAllWindows()
            exit(1)
            
    return crop_box

def sync_video(video_path, telemetry_csv_path, output_dir, crop_box=None):
    print(f"Processing video: {video_path}")
    
    if crop_box:
        print(f"Will crop frames to {crop_box} (x, y, w, h) before resizing.")
        
    # Extract approximate start time from filename
    filename = os.path.basename(video_path)
    # Expected format: 20260710_054604000_iOS.MOV
    date_str = filename.split('_')[0]
    time_str = filename.split('_')[1][:6]
    
    dt = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
    dt = dt.replace(tzinfo=timezone.utc)
    approx_start_ms = int(dt.timestamp() * 1000)
    print(f"Approximate start timestamp: {approx_start_ms}")
    
    # Load Telemetry
    print(f"Loading telemetry from {telemetry_csv_path}...")
    df = pd.read_csv(telemetry_csv_path)
    
    # Open Video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video FPS: {fps}, Total Frames: {total_frames}")
    
    # Detect the first flash
    print("Scanning video for the first visual sync flash...")
    brightness_history = []
    first_flash_frame = -1
    
    for i in range(min(500, total_frames)):
        ret, frame = cap.read()
        if not ret:
            break
            
        # The laptop screen is a large portion. Calculate global mean brightness
        mean_brightness = np.mean(frame)
        brightness_history.append(mean_brightness)
        
        if len(brightness_history) > 5:
            # Detect a jump in brightness (black to dark gray or vice versa)
            diff = abs(brightness_history[-1] - brightness_history[-2])
            if diff > 0.6: # Lowered threshold based on data (typical delta is ~1.2)
                first_flash_frame = i
                print(f"Detected first flash at frame {first_flash_frame} (Brightness delta: {diff:.2f})")
                break
                
    if first_flash_frame == -1:
        print("ERROR: Could not detect any screen flashes in the first 500 frames!")
        return
        
    # We found the first flash! This corresponds to a 500ms boundary in the telemetry.
    # We find the nearest 500ms boundary to our approx start time
    flash_offset_ms = int((first_flash_frame / fps) * 1000)
    expected_flash_time_ms = approx_start_ms + flash_offset_ms
    
    # Round to nearest 500ms boundary
    nearest_500ms = round(expected_flash_time_ms / 500.0) * 500
    print(f"Anchoring frame {first_flash_frame} to exact timestamp {nearest_500ms}")
    
    # Calculate exact start time of frame 0
    exact_start_time_ms = nearest_500ms - flash_offset_ms
    print(f"Exact video start timestamp: {exact_start_time_ms}")
    
    # Reset video
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Setup outputs
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    labels_path = os.path.join(output_dir, 'labels.csv')
    
    # If labels.csv doesn't exist, create it with headers
    write_header = not os.path.exists(labels_path)
    
    with open(labels_path, 'a', newline='') as f:
        # Match columns with telemetry plus the image filename
        header = ['image_file'] + list(df.columns)
        if write_header:
            f.write(','.join(header) + '\n')
            
        print("Extracting and matching frames...")
        frames_saved = 0
        
        # To make it fast, we can extract every Nth frame, or all of them. Let's do all.
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_time_ms = exact_start_time_ms + int((i / fps) * 1000)
            
            # Find closest telemetry row
            # Using searchsorted for speed
            idx = df['host_timestamp_ms'].searchsorted(frame_time_ms)
            
            if idx == 0:
                closest_idx = 0
            elif idx == len(df):
                closest_idx = len(df) - 1
            else:
                before = df['host_timestamp_ms'].iloc[idx - 1]
                after = df['host_timestamp_ms'].iloc[idx]
                if frame_time_ms - before < after - frame_time_ms:
                    closest_idx = idx - 1
                else:
                    closest_idx = idx
                    
            closest_row = df.iloc[closest_idx]
            
            # Only save if the telemetry is reasonably close (e.g. within 50ms)
            time_diff = abs(closest_row['host_timestamp_ms'] - frame_time_ms)
            if time_diff < 50:
                img_name = f"{frame_time_ms}.jpg"
                img_path = os.path.join(output_dir, 'images', img_name)
                
                # Crop the image if a crop box was provided
                if crop_box is not None:
                    cx, cy, cw, ch = crop_box
                    # Ensure we don't go out of bounds
                    fh, fw = frame.shape[:2]
                    cy = max(0, cy)
                    cx = max(0, cx)
                    ch = min(fh - cy, ch)
                    cw = min(fw - cx, cw)
                    frame = frame[cy:cy+ch, cx:cx+cw]
                
                # Resize image for the vision model to save space (e.g. 640x480)
                frame_resized = cv2.resize(frame, (640, 480))
                cv2.imwrite(img_path, frame_resized)
                
                # Write to labels.csv
                row_data = [img_name] + [str(x) for x in closest_row.values]
                f.write(','.join(row_data) + '\n')
                frames_saved += 1
                
            if i % 100 == 0:
                print(f"Processed {i}/{total_frames} frames... ({frames_saved} saved)")
                
    cap.release()
    print(f"Done! Saved {frames_saved} synchronized frames to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--telemetry', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--crop', type=str, help='Crop box as x,y,w,h (e.g. 100,200,800,600)')
    parser.add_argument('--auto-crop', action='store_true', help='Automatically detect and crop the platform')
    parser.add_argument('--padding', type=int, default=50, help='Padding pixels for auto-crop (default 50)')
    args = parser.parse_args()
    
    crop_box = None
    if args.auto_crop:
        crop_box = auto_detect_crop(args.video, num_frames=100, padding=args.padding)
    elif args.crop:
        try:
            crop_box = tuple(map(int, args.crop.split(',')))
            if len(crop_box) != 4:
                raise ValueError("Must provide 4 integers")
        except Exception as e:
            print(f"Error parsing crop argument: {e}")
            exit(1)
            
    sync_video(args.video, args.telemetry, args.output, crop_box)
