import cv2
import numpy as np
import pandas as pd
import os
import argparse
from pathlib import Path

# Physical Marker Coordinates in Platform-Centric Millimeters
# Origin is Top-Left of the platform
PLATFORM_W = 187.5
PLATFORM_H = 142.0

MARKERS_PHYSICAL_MM = [
    (33.0, 26.0),                   # Green (Top-Left)
    (187.5 - 41.0, 53.0),           # Red (Top-Right)
    (187.5 - 13.0, 142.0 - 8.0),    # Black (Bottom-Right)
    (69.0, 142.0 - 58.0)            # Grey (Bottom-Left)
]

clicked_points = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        print(f"Clicked point {len(clicked_points)}: ({x}, {y})")

def main():
    parser = argparse.ArgumentParser(description="Generate YOLO bounding box labels using Homography")
    parser.add_argument("--video", required=True, help="Path to input video (e.g., in 02_silver)")
    parser.add_argument("--telemetry", required=True, help="Path to the synced telemetry CSV")
    parser.add_argument("--output_labels", required=True, help="Directory to save YOLO .txt labels")
    parser.add_argument("--output_images", required=True, help="Directory to save training images")
    args = parser.parse_args()

    # Load telemetry
    df = pd.read_csv(args.telemetry)
    
    # Open video
    cap = cv2.VideoCapture(args.video)
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read video.")
        return

    # 1. Manual Calibration (One-Time per video)
    print("Please click the 4 markers in order: Green (TL), Red (TR), Black (BR), Grey (BL)")
    cv2.imshow("Calibration", frame)
    cv2.setMouseCallback("Calibration", mouse_callback)

    while len(clicked_points) < 4:
        cv2.waitKey(10)

    cv2.destroyAllWindows()
    
    # 2. Compute Homography (Pixels -> Millimeters)
    src_pixels = np.array(clicked_points, dtype=np.float32)
    dst_mm = np.array(MARKERS_PHYSICAL_MM, dtype=np.float32)
    
    # M maps Pixels -> MM
    M, _ = cv2.findHomography(src_pixels, dst_mm)
    
    # M_inv maps MM -> Pixels (we need this to generate the ball bounding box)
    M_inv = np.linalg.inv(M)

    # 3. Generate YOLO Dataset
    os.makedirs(args.output_labels, exist_ok=True)
    os.makedirs(args.output_images, exist_ok=True)

    frame_idx = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    print("Generating YOLO labels for all frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        h, w = frame.shape[:2]
        
        # We need the ball's (touch_x, touch_y) in MM from the telemetry
        if frame_idx >= len(df):
            break
            
        # Assuming telemetry has 'touch_x' and 'touch_y' columns
        # Convert origin from center (used in control) back to Top-Left if necessary
        # The telemetry usually records (0,0) as the center. 
        # So Top-Left origin X = touch_x + (187.5 / 2)
        # Top-Left origin Y = touch_y + (142 / 2)
        
        row = df.iloc[frame_idx]
        ball_x_mm = row['touch_x'] + (PLATFORM_W / 2.0)
        ball_y_mm = row['touch_y'] + (PLATFORM_H / 2.0)
        
        # Project Ball MM to Pixels
        ball_pt_mm = np.array([[[ball_x_mm, ball_y_mm]]], dtype=np.float32)
        ball_pt_pixel = cv2.perspectiveTransform(ball_pt_mm, M_inv)[0][0]
        bx, by = ball_pt_pixel[0], ball_pt_pixel[1]
        
        # YOLO Format: class_id center_x center_y width height (normalized 0-1)
        # Class 0: Ball
        bx_norm = max(0, min(1, bx / w))
        by_norm = max(0, min(1, by / h))
        
        # Assume ball is roughly 30x30 pixels (normalized)
        bw_norm = 30.0 / w
        bh_norm = 30.0 / h
        
        label_lines = []
        label_lines.append(f"0 {bx_norm:.6f} {by_norm:.6f} {bw_norm:.6f} {bh_norm:.6f}")
        
        # Class 1: Markers (we use the 4 clicked points for the whole video since the crop is mostly static)
        # Or even better, we can assume the platform might jiggle a bit if the video isn't perfectly static?
        # If the video is a fixed camera mount, the marker pixels are fixed.
        for mx, my in clicked_points:
            mx_norm = mx / w
            my_norm = my / h
            mw_norm = 20.0 / w
            mh_norm = 20.0 / h
            label_lines.append(f"1 {mx_norm:.6f} {my_norm:.6f} {mw_norm:.6f} {mh_norm:.6f}")
            
        # Save image and label
        base_name = f"frame_{frame_idx:06d}"
        img_path = os.path.join(args.output_images, f"{base_name}.jpg")
        txt_path = os.path.join(args.output_labels, f"{base_name}.txt")
        
        cv2.imwrite(img_path, frame)
        with open(txt_path, 'w') as f:
            f.write('\n'.join(label_lines) + '\n')
            
        frame_idx += 1
        
        if frame_idx % 1000 == 0:
            print(f"Processed {frame_idx} frames...")
            
    cap.release()
    print(f"Done! Generated {frame_idx} labeled images.")

if __name__ == '__main__':
    main()
