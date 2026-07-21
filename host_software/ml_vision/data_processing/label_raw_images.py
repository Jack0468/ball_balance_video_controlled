import cv2
import os
import glob
import random
import numpy as np
import argparse

def get_random_frames(video_path, num_frames=10):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open {video_path}")
        return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < num_frames:
        num_frames = total_frames
        
    frame_indices = sorted(random.sample(range(total_frames), num_frames))
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames

# Global variables for OpenCV mouse callback
current_image = None
display_image = None
clicked_points = []
window_name = "Labeling"

def mouse_callback(event, x, y, flags, param):
    global display_image, clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        # Draw the point
        color = (0, 255, 0) if len(clicked_points) <= 4 else (0, 0, 255)
        cv2.circle(display_image, (x, y), 5, color, -1)
        
        # Connect the corners if we have them
        if len(clicked_points) >= 2 and len(clicked_points) <= 4:
            cv2.line(display_image, clicked_points[-2], clicked_points[-1], (0, 255, 0), 2)
        if len(clicked_points) == 4:
            cv2.line(display_image, clicked_points[3], clicked_points[0], (0, 255, 0), 2)
            
        cv2.imshow(window_name, display_image)

def format_yolo_pose(class_id, cx, cy, w, h, img_w, img_h, keypoints=None):
    cx_n = max(0.0, min(1.0, cx / img_w))
    cy_n = max(0.0, min(1.0, cy / img_h))
    w_n = max(0.0, min(1.0, w / img_w))
    h_n = max(0.0, min(1.0, h / img_h))
    
    label = f"{class_id} {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}"
    
    if keypoints is not None:
        for (kx, ky) in keypoints:
            kx_n = max(0.0, min(1.0, kx / img_w))
            ky_n = max(0.0, min(1.0, ky / img_h))
            label += f" {kx_n:.6f} {ky_n:.6f} 2"
    else:
        for _ in range(4):
            label += " 0.000000 0.000000 0"
            
    return label

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bronze-dir', default='host_software/ml_vision/data/01_bronze', help='Dir containing video folders')
    parser.add_argument('--out-dir', default='host_software/ml_vision/data/yolo_raw_dataset', help='Output dataset dir')
    parser.add_argument('--frames-per-video', type=int, default=10, help='Frames to extract per video')
    args = parser.parse_args()
    
    images_dir = os.path.join(args.out_dir, 'images')
    labels_dir = os.path.join(args.out_dir, 'labels')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    # 1. Extract frames if not already done
    existing_images = glob.glob(os.path.join(images_dir, '*.jpg'))
    if len(existing_images) == 0:
        print("Extracting random frames from videos...")
        video_paths = glob.glob(os.path.join(args.bronze_dir, '**', '*.MOV'), recursive=True)
        if not video_paths:
            print("No videos found!")
            return
            
        img_id = 0
        for vp in video_paths:
            print(f"Extracting from {vp}...")
            frames = get_random_frames(vp, args.frames_per_video)
            for f in frames:
                cv2.imwrite(os.path.join(images_dir, f"{img_id:04d}.jpg"), f)
                img_id += 1
                
        existing_images = glob.glob(os.path.join(images_dir, '*.jpg'))
    else:
        print(f"Found {len(existing_images)} existing images in {images_dir}.")

    # 2. Labeling Loop
    global display_image, clicked_points, current_image
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    for img_path in sorted(existing_images):
        txt_path = img_path.replace('images', 'labels').replace('.jpg', '.txt')
        if os.path.exists(txt_path):
            continue # Already labeled
            
        current_image = cv2.imread(img_path)
        if current_image is None:
            continue
            
        img_h, img_w = current_image.shape[:2]
        
        while True:
            display_image = current_image.copy()
            clicked_points = []
            
            # Instructions
            cv2.putText(display_image, "Click 4 corners (TL, TR, BR, BL), then 1 for Ball Center.", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(display_image, "Press 'c' to clear, 's' to save/skip.", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow(window_name, display_image)
            cv2.setMouseCallback(window_name, mouse_callback)
            
            key = cv2.waitKey(0) & 0xFF
            if key == ord('c'):
                continue # Reset
            elif key == ord('s'):
                if len(clicked_points) == 5:
                    # Save
                    corners = clicked_points[:4]
                    ball_center = clicked_points[4]
                    
                    # Compute board bbox
                    xs = [p[0] for p in corners]
                    ys = [p[1] for p in corners]
                    plat_x_min, plat_x_max = min(xs), max(xs)
                    plat_y_min, plat_y_max = min(ys), max(ys)
                    plat_w = plat_x_max - plat_x_min
                    plat_h = plat_y_max - plat_y_min
                    plat_cx = plat_x_min + plat_w / 2.0
                    plat_cy = plat_y_min + plat_h / 2.0
                    
                    label0 = format_yolo_pose(0, plat_cx, plat_cy, plat_w, plat_h, img_w, img_h, keypoints=corners)
                    
                    ball_w = 40.0
                    ball_h = 40.0
                    label1 = format_yolo_pose(1, ball_center[0], ball_center[1], ball_w, ball_h, img_w, img_h, keypoints=None)
                    
                    with open(txt_path, 'w') as f:
                        f.write(label0 + '\n')
                        f.write(label1 + '\n')
                        
                    print(f"Saved {txt_path}")
                    break
                else:
                    print("You must click exactly 5 points (4 corners + 1 ball)! Try again.")
                    continue
            elif key == 27: # ESC
                print("Exiting...")
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()
    print("All images labeled!")

if __name__ == '__main__':
    main()
