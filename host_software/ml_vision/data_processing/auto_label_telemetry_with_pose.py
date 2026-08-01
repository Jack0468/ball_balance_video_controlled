import cv2
import numpy as np
import pandas as pd
import os
import argparse
from ultralytics import YOLO
from tqdm import tqdm

# Physical dimensions
TOUCHPAD_W = 187.5
TOUCHPAD_H = 142.0
PAPER_W = 164.0
PAPER_H = 124.0

# As calculated before, the paper is centered on the touchpad
OFFSET_X = (TOUCHPAD_W - PAPER_W) / 2.0
OFFSET_Y = (TOUCHPAD_H - PAPER_H) / 2.0

# 4 Corners of the Paper in Touchpad MM coordinate space (0,0 is top-left of touchpad)
# IMPORTANT: YOLO learned these keypoints with a Y-flip applied (from generate_pose_dataset_from_aruco.py).
# We must define them in the exact same order (0, 1, 2, 3) and values so findHomography matches them correctly.
PAPER_CORNERS_MM = np.array([
    [OFFSET_X, TOUCHPAD_H - OFFSET_Y],                     
    [OFFSET_X + PAPER_W, TOUCHPAD_H - OFFSET_Y],             
    [OFFSET_X + PAPER_W, TOUCHPAD_H - (OFFSET_Y + PAPER_H)],     
    [OFFSET_X, TOUCHPAD_H - (OFFSET_Y + PAPER_H)]              
], dtype=np.float32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="host_software/ml_vision/models/yolov8_platform_corners_v1/weights/best.pt")
    parser.add_argument("--data_dir", default="host_software/data/02_silver")
    args = parser.parse_args()
    
    # Normally we would check if the model exists here, but we'll let it fail loud if not since it's just a script
    print(f"Loading trained pose model: {args.model}")
    try:
        model = YOLO(args.model)
    except Exception as e:
        print(f"Could not load model, error: {e}")
        return
        
    sessions = [d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))]
    
    if "images_iphone" in sessions:
        sessions.remove("images_iphone")

    for session in sessions:
        session_dir = os.path.join(args.data_dir, session)
        csv_path = os.path.join(session_dir, "labels_normalized.csv")
        
        if not os.path.exists(csv_path):
            print(f"Skipping {session}: no labels_normalized.csv found.")
            continue
            
        print(f"Processing session: {session}...")
        df = pd.read_csv(csv_path)
        
        # We will create a new CSV and a new cropped directory
        out_csv_path = os.path.join(session_dir, "yolo_platform_corners_features.csv")
        # Remove existing partial CSV if it exists so we start fresh for this session
        if os.path.exists(out_csv_path):
            os.remove(out_csv_path)
            
        out_crop_dir = os.path.join(session_dir, "cropped_pose", "images")
        os.makedirs(out_crop_dir, exist_ok=True)
        
        results_list = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Auto-labeling {session}"):
            img_name = row['image_file']
            img_path = os.path.join(session_dir, "images", img_name)
            
            if not os.path.exists(img_path):
                continue
                
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            # 1. Run YOLO-Pose to find the 4 corners of the paper
            results = model.predict(source=img, imgsz=640, conf=0.5, verbose=False)
            result = results[0]
            
            if result.keypoints is None or len(result.keypoints.xy) == 0:
                continue
                
            kpts = result.keypoints.xy[0].cpu().numpy()
            if len(kpts) < 4:
                continue
                
            # Use YOLO's native keypoint order! It already knows which corner is which.
            kpts = kpts[:4]
            
            # 2. Compute Homography mapping Touchpad MM -> Image Pixels
            M_inv, _ = cv2.findHomography(PAPER_CORNERS_MM, kpts)
            if M_inv is None:
                continue
                
            # 3. Telemetry is relative to the center of the touchpad.
            # Convert to Touchpad MM space (where Top-Left is 0,0)
            # NOTE: physical touch_y increases UP, but image Y increases DOWN.
            # So we must subtract touch_y from the center instead of adding it.
            touch_x = row['touch_x']
            touch_y = row['touch_y']
            
            ball_x_mm = touch_x + (TOUCHPAD_W / 2.0)
            ball_y_mm = (TOUCHPAD_H / 2.0) - touch_y
            
            ball_pt_mm = np.array([[[ball_x_mm, ball_y_mm]]], dtype=np.float32)
            ball_pt_pixel = cv2.perspectiveTransform(ball_pt_mm, M_inv)[0][0]
            
            # Generate bounding box for CNN
            bw, bh = 30.0, 30.0 # Bounding box size in pixels
            x_min = ball_pt_pixel[0] - (bw / 2.0)
            y_min = ball_pt_pixel[1] - (bh / 2.0)
            
            row_dict = row.to_dict()
            row_dict['ball_x'] = int(ball_pt_pixel[0])
            row_dict['ball_y'] = int(ball_pt_pixel[1])
            row_dict['ball_xmin'] = int(x_min)
            row_dict['ball_ymin'] = int(y_min)
            row_dict['ball_xmax'] = int(x_min + bw)
            row_dict['ball_ymax'] = int(y_min + bh)
            
            # 4. Generate the perfect cropped image using M
            scale = 2.0
            crop_w = int(TOUCHPAD_W * scale)
            crop_h = int(TOUCHPAD_H * scale)
            
            dst_pts = np.array([
                [0, 0],
                [crop_w, 0],
                [crop_w, crop_h],
                [0, crop_h]
            ], dtype=np.float32)
            
            touchpad_corners_mm = np.array([
                [0, 0],
                [TOUCHPAD_W, 0],
                [TOUCHPAD_W, TOUCHPAD_H],
                [0, TOUCHPAD_H]
            ], dtype=np.float32)
            
            touchpad_pixel_corners = cv2.perspectiveTransform(np.array([touchpad_corners_mm]), M_inv)[0]
            
            M_crop, _ = cv2.findHomography(touchpad_pixel_corners, dst_pts)
            
            if M_crop is not None:
                img_cropped = cv2.warpPerspective(img, M_crop, (crop_w, crop_h))
                cv2.imwrite(os.path.join(out_crop_dir, img_name), img_cropped)
                
                ball_pt_cropped = cv2.perspectiveTransform(np.array([[ball_pt_pixel]]), M_crop)[0][0]
                row_dict['crop_ball_x'] = int(ball_pt_cropped[0])
                row_dict['crop_ball_y'] = int(ball_pt_cropped[1])
                row_dict['crop_w'] = crop_w
                row_dict['crop_h'] = crop_h
                
            results_list.append(row_dict)
            
            # Periodically save to disk to save RAM
            if len(results_list) >= 100:
                out_df = pd.DataFrame(results_list)
                out_df.to_csv(out_csv_path, mode='a', header=not os.path.exists(out_csv_path), index=False)
                results_list = []
            
        # Save any remaining results
        if len(results_list) > 0:
            out_df = pd.DataFrame(results_list)
            out_df.to_csv(out_csv_path, mode='a', header=not os.path.exists(out_csv_path), index=False)
            results_list = []
            
        print(f"Finished auto-labeling {session} -> {out_csv_path}")
            
if __name__ == "__main__":
    main()
