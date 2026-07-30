import cv2
import numpy as np
import pandas as pd
import os
import argparse
import cv2.aruco as aruco

# Physical Platform Dimensions
PLATFORM_W = 187.5
PLATFORM_H = 142.0
CROP_PAD = 20

# These are the exact millimeter coordinates of the centers of the 6 markers
# relative to the Top-Left (0,0) corner of the printed PDF bounding box.
# Note: The physical setup requires a vertical flip (only Y is inverted).
# So we keep original X, but subtract Y from PLATFORM_H (142.0).
MARKER_PHYSICAL_MM = {
    0: [12.0, 130.0],
    1: [175.5, 130.0],
    2: [175.5, 12.0],
    3: [12.0, 12.0],
    4: [12.0, 71.0],
    5: [175.5, 71.0]
}

PLATFORM_CORNERS_MM = np.array([
    [[0.0, 0.0]],
    [[PLATFORM_W, 0.0]],
    [[PLATFORM_W, PLATFORM_H]],
    [[0.0, PLATFORM_H]]
], dtype=np.float32)

# Global variables for mouse callback
clicked_point = None

def mouse_callback(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)

def main():
    parser = argparse.ArgumentParser(description="Generate ArUco cropped dataset")
    parser.add_argument("--video", required=True, help="Path to rgb_video.mp4 from 01_bronze")
    parser.add_argument("--out_dir", required=True, help="Directory to save the new dataset (e.g., 02_silver/session_X)")
    args = parser.parse_args()

    # Create output directories
    os.makedirs(args.out_dir, exist_ok=True)
    images_dir = os.path.join(args.out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Init ArUco
    try:
        dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        parameters = aruco.DetectorParameters()
        detector = aruco.ArucoDetector(dictionary, parameters)
    except AttributeError:
        dictionary = aruco.Dictionary_get(aruco.DICT_4X4_50)
        parameters = aruco.DetectorParameters_create()
        detector = None

    video_path = os.path.abspath(args.video)
    bronze_dir = os.path.dirname(video_path)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Failed to open video")
        return
        
    telemetry_path = os.path.join(bronze_dir, 'telemetry.csv')
    timestamps_path = os.path.join(bronze_dir, 'frame_timestamps.csv')
    
    synced_telemetry = {}
    if os.path.exists(telemetry_path) and os.path.exists(timestamps_path):
        print(f"Loading telemetry from {bronze_dir}...")
        df_tel = pd.read_csv(telemetry_path).sort_values('host_timestamp_ms')
        df_ts = pd.read_csv(timestamps_path).sort_values('frame_timestamp_ms')
        
        merged = pd.merge_asof(
            df_ts, df_tel,
            left_on='frame_timestamp_ms',
            right_on='host_timestamp_ms',
            direction='nearest'
        )
        
        for _, row in merged.iterrows():
            f_idx = int(row['frame_index'])
            synced_telemetry[f_idx] = {
                'touch_x': row['touch_x'],
                'touch_y': row['touch_y']
            }
        print(f"Successfully synced {len(synced_telemetry)} frames with touch pad telemetry!")
    else:
        print("Warning: telemetry.csv or frame_timestamps.csv not found in the video folder. No touch data will be saved.")

    csv_path = os.path.join(args.out_dir, 'yolo_features.csv')
    
    features = []
    frame_idx = 0
    saved_count = 0
    
    # State tracking to detect if ball fell off (frozen touch coordinates)
    prev_touch_x = None
    prev_touch_y = None
    
    # If appending, figure out what the next saved_count should be
    if os.path.exists(csv_path):
        try:
            existing_df = pd.read_csv(csv_path)
            if not existing_df.empty:
                # Extract the highest number from "frame_XXXX.jpg"
                last_frame = existing_df['image_file'].iloc[-1]
                saved_count = int(last_frame.replace("frame_", "").replace(".jpg", "")) + 1
                print(f"Found existing dataset with {len(existing_df)} entries. Appending starting from frame_{saved_count:04d}.jpg")
        except Exception as e:
            print(f"Error reading existing CSV: {e}")

    cv2.namedWindow("Label Ball")
    cv2.setMouseCallback("Label Ball", mouse_callback)

    print("Instructions:")
    print(" - Click the center of the ball in the cropped image.")
    print(" - Press SPACE to skip a frame (if no ball or blurry).")
    print(" - Press 'q' to quit and save the dataset.")

    global clicked_point

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        
        # To speed up labeling, we can skip every N frames (e.g., sample at 5 fps instead of 30)
        # Let's extract 1 frame every 6 frames
        if frame_idx % 6 != 0:
            continue

        h_frame, w_frame = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Detect ArUco
        if detector is not None:
            corners, ids, rejected = detector.detectMarkers(gray)
        else:
            corners, ids, rejected = aruco.detectMarkers(gray, dictionary, parameters=parameters)

        if ids is None:
            continue
            
        ids = ids.flatten()
        pixel_centers = []
        physical_centers = []
        
        for i, marker_id in enumerate(ids):
            if marker_id in MARKER_PHYSICAL_MM:
                marker_corners = corners[i][0]
                center = np.mean(marker_corners, axis=0)
                pixel_centers.append(center)
                physical_centers.append(MARKER_PHYSICAL_MM[marker_id])
        
        if len(pixel_centers) < 4:
            continue
            
        pixel_centers = np.array(pixel_centers, dtype=np.float32)
        physical_centers = np.array(physical_centers, dtype=np.float32)
        M, _ = cv2.findHomography(pixel_centers, physical_centers)
        
        if M is None:
            continue
            
        # 2. Crop exactly like main_aruco.py
        M_inv = np.linalg.inv(M)
        pixel_corners = cv2.perspectiveTransform(PLATFORM_CORNERS_MM, M_inv)
        
        xs = pixel_corners[:, 0, 0]
        ys = pixel_corners[:, 0, 1]
        x1 = int(max(0,        xs.min() - CROP_PAD))
        y1 = int(max(0,        ys.min() - CROP_PAD))
        x2 = int(min(w_frame,  xs.max() + CROP_PAD))
        y2 = int(min(h_frame,  ys.max() + CROP_PAD))
        
        if x2 <= x1 or y2 <= y1:
            continue
            
        crop = frame[y1:y2, x1:x2]
        
        # 3. Automatically save the crop and calculate pixel labels using Inverse Homography
        current_frame_index = frame_idx - 1
        if current_frame_index in synced_telemetry:
            # The telemetry from the firmware is centered at (0,0) (i.e. Center of the board)
            # But the ArUco Homography matrix M uses the Top-Left corner as (0,0)!
            # We must shift the telemetry by half the platform width/height to match the ArUco origin!
            touch_x_centered = synced_telemetry[current_frame_index]['touch_x']
            touch_y_centered = synced_telemetry[current_frame_index]['touch_y']
            
            # Heuristic to detect if the ball has fallen off the board or is teleporting!
            # The firmware holds touch_x/y perfectly constant when pressure (z) drops to 0.
            if prev_touch_x is not None:
                dist = ((touch_x_centered - prev_touch_x)**2 + (touch_y_centered - prev_touch_y)**2)**0.5
                if dist == 0.0 or dist > 30.0:
                    prev_touch_x = touch_x_centered
                    prev_touch_y = touch_y_centered
                    continue
            
            prev_touch_x = touch_x_centered
            prev_touch_y = touch_y_centered
            
            touch_x_topleft = touch_x_centered + (PLATFORM_W / 2.0)
            touch_y_topleft = touch_y_centered + (PLATFORM_H / 2.0)
            
            # Use Inverse Homography to map physical telemetry back to full-frame pixels
            touch_pt = np.array([[[touch_x_topleft, touch_y_topleft]]], dtype=np.float32)
            frame_pt = cv2.perspectiveTransform(touch_pt, M_inv)
            ball_frame_x = float(frame_pt[0, 0, 0])
            ball_frame_y = float(frame_pt[0, 0, 1])
            
            # Subtract the bounding box offsets to get crop-relative pixels
            ball_x = ball_frame_x - x1
            ball_y = ball_frame_y - y1
            
            img_name = f"frame_{saved_count:04d}.jpg"
            cv2.imwrite(os.path.join(images_dir, img_name), crop)
            
            feat = {
                'image_file': img_name,
                'ball_present': 1.0,
                'touch_x': touch_x_centered,
                'touch_y': touch_y_centered,
                'ball_x': ball_x,
                'ball_y': ball_y
            }
            features.append(feat)
            saved_count += 1
            
            if saved_count % 100 == 0:
                print(f"Automatically extracted and labeled {saved_count} frames...")

    cap.release()
    cv2.destroyAllWindows()
    
    # Save CSV
    if features:
        new_df = pd.DataFrame(features)
        if os.path.exists(csv_path):
            existing_df = pd.read_csv(csv_path)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df.to_csv(csv_path, index=False)
        else:
            new_df.to_csv(csv_path, index=False)
        print(f"\nAppended {len(features)} new labeled crops to {csv_path}!")
    else:
        print("\nNo frames labeled.")

if __name__ == '__main__':
    main()
