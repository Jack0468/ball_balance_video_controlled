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
marker_points = [] # list of tuples (class_id, x, y)
current_marker_class = None
window_name = "Labeling"
existing_labels = []

def mouse_callback(event, x, y, flags, param):
    global display_image, clicked_points, marker_points, current_marker_class
    if event == cv2.EVENT_LBUTTONDOWN:
        if current_marker_class is not None:
            marker_points.append((current_marker_class, x, y))
            current_marker_class = None # Reset after one click
        else:
            if len(clicked_points) < 5:
                clicked_points.append((x, y))
        redraw()

def redraw():
    global display_image, current_image, clicked_points, marker_points, existing_labels
    display_image = current_image.copy()
    
    img_h, img_w = display_image.shape[:2]

    # Draw existing labels
    for label_str in existing_labels:
        parts = label_str.strip().split()
        if len(parts) >= 5:
            c_id = int(parts[0])
            cx = float(parts[1]) * img_w
            cy = float(parts[2]) * img_h
            
            if c_id == 0:
                cv2.putText(display_image, "Platform", (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                # optionally draw keypoints if present
                if len(parts) >= 17:
                    pts = []
                    for i in range(4):
                        kx = int(float(parts[5 + i*3]) * img_w)
                        ky = int(float(parts[6 + i*3]) * img_h)
                        v = int(float(parts[7 + i*3]))
                        if v > 0:
                            pts.append((kx, ky))
                            cv2.circle(display_image, (kx, ky), 5, (0, 255, 0), -1)
                    if len(pts) == 4:
                        for i in range(4):
                            cv2.line(display_image, pts[i], pts[(i+1)%4], (0, 255, 0), 2)
            elif c_id == 1:
                cv2.circle(display_image, (int(cx), int(cy)), 10, (0, 0, 255), -1)
                cv2.putText(display_image, "Ball", (int(cx), int(cy)-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            elif c_id >= 2:
                colors = {2:(255,0,0), 3:(128,128,128), 4:(0,0,0), 5:(0,0,255), 6:(0,255,0), 7:(0,255,255), 8:(255,255,0), 9:(255,0,255), 10:(0,165,255), 11:(203,192,255), 12:(42,42,165)}
                names = {2:"Blue", 3:"Grey", 4:"Black", 5:"Red", 6:"Green", 7:"Yellow", 8:"Cyan", 9:"Purple", 10:"Orange", 11:"Pink", 12:"Brown"}
                cv2.circle(display_image, (int(cx), int(cy)), 8, colors.get(c_id, (255,255,255)), -1)
                cv2.putText(display_image, names.get(c_id, ""), (int(cx), int(cy)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors.get(c_id, (255,255,255)), 1)
                
    
    # Draw new points
    for i, p in enumerate(clicked_points):
        color = (0, 255, 0) if i < 4 else (0, 0, 255)
        cv2.circle(display_image, p, 5, color, -1)
        
    if len(clicked_points) >= 2 and len(clicked_points) <= 4:
        cv2.line(display_image, clicked_points[-2], clicked_points[-1], (0, 255, 0), 2)
    if len(clicked_points) == 4:
        cv2.line(display_image, clicked_points[3], clicked_points[0], (0, 255, 0), 2)
        
    # Draw new markers
    colors = {2:(255,0,0), 3:(128,128,128), 4:(0,0,0), 5:(0,0,255), 6:(0,255,0), 7:(0,255,255), 8:(255,255,0), 9:(255,0,255), 10:(0,165,255), 11:(203,192,255), 12:(42,42,165)}
    for (c_id, x, y) in marker_points:
        cv2.circle(display_image, (x, y), 8, colors.get(c_id, (255,255,255)), -1)
        
    # Instructions
    cv2.putText(display_image, "If new frame: Click 4 corners (TL, TR, BR, BL), then 1 for Ball.", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(display_image, "Markers: 2(Blu),3(Gry),4(Blk),5(Red),6(Grn),7(Yel),8(Cya),9(Pur),o(Org),p(Pnk),b(Brn)", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(display_image, "Press 's' to save/next, 'c' to clear new points, 'x' to skip frame.", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
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
    global display_image, clicked_points, marker_points, current_image, current_marker_class, existing_labels
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    for img_path in sorted(existing_images):
        txt_path = img_path.replace('images', 'labels').replace('.jpg', '.txt')
        
        existing_labels = []
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                existing_labels = f.readlines()
                
        current_image = cv2.imread(img_path)
        if current_image is None:
            continue
            
        img_h, img_w = current_image.shape[:2]
        
        while True:
            clicked_points = []
            marker_points = []
            current_marker_class = None
            redraw()
            
            while True:
                cv2.setMouseCallback(window_name, mouse_callback)
                key = cv2.waitKey(0) & 0xFF
                
                if key == ord('c'):
                    clicked_points = []
                    marker_points = []
                    current_marker_class = None
                    redraw()
                elif key == ord('x'):
                    print(f"Skipping {img_path}")
                    break
                elif key == ord('2'):
                    current_marker_class = 2
                    print("Click Blue Marker")
                elif key == ord('3'):
                    current_marker_class = 3
                    print("Click Grey Marker")
                elif key == ord('4'):
                    current_marker_class = 4
                    print("Click Black Marker")
                elif key == ord('5'):
                    current_marker_class = 5
                    print("Click Red Marker")
                elif key == ord('6'):
                    current_marker_class = 6
                    print("Click Green Marker")
                elif key == ord('7'):
                    current_marker_class = 7
                    print("Click Yellow Marker")
                elif key == ord('8'):
                    current_marker_class = 8
                    print("Click Cyan Marker")
                elif key == ord('9'):
                    current_marker_class = 9
                    print("Click Purple Marker")
                elif key == ord('o'):
                    current_marker_class = 10
                    print("Click Orange Marker")
                elif key == ord('p'):
                    current_marker_class = 11
                    print("Click Pink Marker")
                elif key == ord('b'):
                    current_marker_class = 12
                    print("Click Brown Marker")
                elif key == ord('s'):
                    # Save
                    new_labels = []
                    # Process new platform/ball if any
                    if len(clicked_points) == 5:
                        corners = clicked_points[:4]
                        ball_center = clicked_points[4]
                        
                        xs = [p[0] for p in corners]
                        ys = [p[1] for p in corners]
                        plat_x_min, plat_x_max = min(xs), max(xs)
                        plat_y_min, plat_y_max = min(ys), max(ys)
                        plat_w = plat_x_max - plat_x_min
                        plat_h = plat_y_max - plat_y_min
                        plat_cx = plat_x_min + plat_w / 2.0
                        plat_cy = plat_y_min + plat_h / 2.0
                        
                        label0 = format_yolo_pose(0, plat_cx, plat_cy, plat_w, plat_h, img_w, img_h, keypoints=corners)
                        label1 = format_yolo_pose(1, ball_center[0], ball_center[1], 40.0, 40.0, img_w, img_h, keypoints=None)
                        new_labels.extend([label0, label1])
                    elif len(clicked_points) > 0:
                        print("Warning: Incomplete clicks for platform/ball. They won't be saved.")
                        
                    # Process markers
                    for (c_id, mx, my) in marker_points:
                        m_label = format_yolo_pose(c_id, mx, my, 20.0, 20.0, img_w, img_h, keypoints=None)
                        new_labels.append(m_label)
                        
                    if not new_labels and not existing_labels:
                        print("Nothing to save, skipping...")
                        break
                        
                    with open(txt_path, 'w') as f:
                        for el in existing_labels:
                            f.write(el.strip() + '\n')
                        for nl in new_labels:
                            f.write(nl + '\n')
                            
                    print(f"Saved {txt_path}")
                    break
                elif key == 27: # ESC
                    print("Exiting...")
                    cv2.destroyAllWindows()
                    return
                    
            break # Move to next image

    cv2.destroyAllWindows()
    print("All images labeled!")

if __name__ == '__main__':
    main()

