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
    parser = argparse.ArgumentParser(description="Generate YOLO labels from existing images and labels.csv")
    parser.add_argument("--data_dir", required=True, help="Path to data directory (e.g., data/02_silver)")
    args = parser.parse_args()

    images_dir = os.path.join(args.data_dir, "images")
    labels_csv_path = os.path.join(args.data_dir, "labels.csv")
    output_labels_dir = os.path.join(args.data_dir, "labels")

    if not os.path.exists(labels_csv_path):
        print(f"Error: Could not find {labels_csv_path}")
        return

    # Load telemetry
    df = pd.read_csv(labels_csv_path)
    if len(df) == 0:
        print("Error: labels.csv is empty")
        return

    # 1. Manual Calibration (One-Time)
    frame_index = len(df) // 2
    frame = None
    while frame_index < len(df):
        first_image_filename = df.iloc[frame_index]['image_file']
        first_image_path = os.path.join(images_dir, first_image_filename)
        
        frame = cv2.imread(first_image_path)
        if frame is None:
            frame_index += 1
            continue

        print("=========================================================")
        print(f"Viewing Frame {frame_index}")
        print("Please click the 4 markers in order:")
        print("1. Green (Top-Left)")
        print("2. Red (Top-Right)")
        print("3. Black (Bottom-Right)")
        print("4. Grey (Bottom-Left)")
        print("--> Press 'n' to skip frame | Press 'c' to cycle contrast modes <--")
        print("=========================================================")
        
        contrast_mode = 0
        display_frame = frame.copy()
        
        clicked_points.clear()

        skip = False
        while len(clicked_points) < 4:
            if contrast_mode == 0:
                display_frame = frame.copy()
            elif contrast_mode == 1:
                # CLAHE (Adaptive Histogram)
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                cl = clahe.apply(l)
                limg = cv2.merge((cl,a,b))
                display_frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            elif contrast_mode == 2:
                # Darken
                display_frame = cv2.convertScaleAbs(frame, alpha=0.5, beta=0)
            elif contrast_mode == 3:
                # Lighten
                display_frame = cv2.convertScaleAbs(frame, alpha=1.5, beta=0)
            elif contrast_mode == 4:
                # Grayscale Histogram Equalization
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                display_frame = cv2.cvtColor(cv2.equalizeHist(gray), cv2.COLOR_GRAY2BGR)
                
            # Draw circles where user already clicked
            for i, pt in enumerate(clicked_points):
                cv2.circle(display_frame, pt, 5, (0, 255, 255), -1)
                
            cv2.imshow("Calibration", display_frame)
            cv2.setMouseCallback("Calibration", mouse_callback)
            
            key = cv2.waitKey(10) & 0xFF
            if key == ord('n'):
                skip = True
                break
            elif key == ord('c'):
                contrast_mode = (contrast_mode + 1) % 5

        if skip:
            print("Skipping to next frame...")
            frame_index += 1
            continue
            
        cv2.destroyAllWindows()
        break
        
    if len(clicked_points) < 4:
        print("Error: Calibration aborted.")
        return
    
    # 2. Compute Homography (Pixels -> Millimeters)
    src_pixels = np.array(clicked_points, dtype=np.float32)
    dst_mm = np.array(MARKERS_PHYSICAL_MM, dtype=np.float32)
    
    M, _ = cv2.findHomography(src_pixels, dst_mm)
    M_inv = np.linalg.inv(M)

    # 3. Generate YOLO Dataset
    os.makedirs(output_labels_dir, exist_ok=True)
    h, w = frame.shape[:2]

    print("Generating YOLO labels for all frames...")
    
    for idx, row in df.iterrows():
        # Get ball (touch_x, touch_y) in MM
        # Telemetry records (0,0) as center. Convert to Top-Left origin:
        ball_x_mm = row['touch_x'] + (PLATFORM_W / 2.0)
        ball_y_mm = row['touch_y'] + (PLATFORM_H / 2.0)
        
        # Project Ball MM to Pixels
        ball_pt_mm = np.array([[[ball_x_mm, ball_y_mm]]], dtype=np.float32)
        ball_pt_pixel = cv2.perspectiveTransform(ball_pt_mm, M_inv)[0][0]
        bx, by = ball_pt_pixel[0], ball_pt_pixel[1]
        
        # YOLO Format: class_id center_x center_y width height (normalized 0-1)
        bx_norm = max(0, min(1, bx / w))
        by_norm = max(0, min(1, by / h))
        bw_norm = 30.0 / w  # Assume 30px width
        bh_norm = 30.0 / h
        
        label_lines = [f"0 {bx_norm:.6f} {by_norm:.6f} {bw_norm:.6f} {bh_norm:.6f}"]
        
        # Add 4 Markers
        for mx, my in clicked_points:
            mx_norm = mx / w
            my_norm = my / h
            mw_norm = 20.0 / w
            mh_norm = 20.0 / h
            label_lines.append(f"1 {mx_norm:.6f} {my_norm:.6f} {mw_norm:.6f} {mh_norm:.6f}")
            
        # Write to .txt file
        txt_filename = row['image_file'].replace('.jpg', '.txt').replace('.png', '.txt')
        txt_path = os.path.join(output_labels_dir, txt_filename)
        
        with open(txt_path, 'w') as f:
            f.write('\n'.join(label_lines) + '\n')
            
        if idx % 1000 == 0 and idx > 0:
            print(f"Processed {idx} images...")
            
    print(f"Done! Generated {len(df)} YOLO .txt labels in {output_labels_dir}.")

if __name__ == '__main__':
    main()
