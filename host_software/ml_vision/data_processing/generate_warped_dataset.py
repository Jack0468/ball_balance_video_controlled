import os
import cv2
import numpy as np
import pandas as pd
import random

# Physical dimensions of the purple board (in mm)
PLATFORM_W = 187.5
PLATFORM_H = 142.0

# The physical corners of the board (Top-Left origin)
BOARD_CORNERS_MM = [
    (0.0, 0.0),                     # Top-Left
    (PLATFORM_W, 0.0),              # Top-Right
    (PLATFORM_W, PLATFORM_H),       # Bottom-Right
    (0.0, PLATFORM_H)               # Bottom-Left
]

clicked_corners = []
clicked_targets = []

def mouse_callback_corners(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_corners.append((x, y))
        print(f"Clicked board corner {len(clicked_corners)}: ({x}, {y})")

def mouse_callback_targets(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_targets.append((x, y))
        print(f"Clicked target marker {len(clicked_targets)}: ({x}, {y})")

def generate_random_warp(img_w, img_h):
    # Base points
    pts1 = np.float32([[0,0], [img_w,0], [img_w,img_h], [0,img_h]])
    
    # Add random jitter to corners to create a perspective warp
    jitter_x = int(img_w * 0.15)
    jitter_y = int(img_h * 0.15)
    
    pts2 = np.float32([
        [random.randint(-jitter_x, jitter_x), random.randint(-jitter_y, jitter_y)],
        [img_w + random.randint(-jitter_x, jitter_x), random.randint(-jitter_y, jitter_y)],
        [img_w + random.randint(-jitter_x, jitter_x), img_h + random.randint(-jitter_x, jitter_x)],
        [random.randint(-jitter_x, jitter_x), img_h + random.randint(-jitter_x, jitter_x)]
    ])
    
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    return matrix

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.abspath(os.path.join(script_dir, "data/02_silver/images"))
    telemetry_path = os.path.abspath(os.path.join(script_dir, "data/02_silver/labels.csv"))
    
    output_dir = os.path.abspath(os.path.join(script_dir, "data/05_warped_dataset"))
    out_images_dir = os.path.join(output_dir, "images")
    out_labels_dir = os.path.join(output_dir, "labels")
    
    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_labels_dir, exist_ok=True)
    
    print("Loading telemetry...")
    df = pd.read_csv(telemetry_path)
    telemetry_dict = {str(row['image_file']): row for _, row in df.iterrows()}
        
    image_files = [f for f in os.listdir(input_dir) if f.endswith('.jpg') or f.endswith('.png')]
    image_files = image_files[:2000]
    
    if not image_files:
        print("No images found!")
        return

    # 1. Manual Calibration on the first image
    first_img_path = os.path.join(input_dir, image_files[0])
    first_img = cv2.imread(first_img_path)
    
    print("STEP 1: Please click the 4 CORNERS of the purple board (Top-Left, Top-Right, Bottom-Right, Bottom-Left)")
    cv2.imshow("Calibration - Board Corners", first_img)
    cv2.setMouseCallback("Calibration - Board Corners", mouse_callback_corners)
    while len(clicked_corners) < 4:
        cv2.waitKey(10)
    cv2.destroyAllWindows()
    
    print("STEP 2: Please click the 4 TARGET MARKERS (Green, Red, Black, Grey)")
    cv2.imshow("Calibration - Target Markers", first_img)
    cv2.setMouseCallback("Calibration - Target Markers", mouse_callback_targets)
    while len(clicked_targets) < 4:
        cv2.waitKey(10)
    cv2.destroyAllWindows()
    
    # Base homography (from real mm to original pixels using the BOARD CORNERS)
    src_pixels = np.array(clicked_corners, dtype=np.float32)
    dst_mm = np.array(BOARD_CORNERS_MM, dtype=np.float32)
    
    # M maps Pixels -> MM
    M, _ = cv2.findHomography(src_pixels, dst_mm)
    M_inv = np.linalg.inv(M) # MM -> Pixels
    
    print(f"Generating labels for {len(image_files)} images...")
    
    success_count = 0
    for idx, img_name in enumerate(image_files):
        img_path = os.path.join(input_dir, img_name)
        if img_name not in telemetry_dict:
            continue
            
        row = telemetry_dict[img_name]
        touch_x = row['touch_x']
        touch_y = row['touch_y']
        
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        
        # Project ball MM to original Pixels
        ball_x_mm = touch_x + (PLATFORM_W / 2.0)
        ball_y_mm = touch_y + (PLATFORM_H / 2.0)
        
        ball_pt_mm = np.array([[[ball_x_mm, ball_y_mm]]], dtype=np.float32)
        ball_pt_pixel = cv2.perspectiveTransform(ball_pt_mm, M_inv)[0][0]
        
        # We now have 5 points in original pixel coordinates:
        # 4 target markers (clicked_targets) and 1 ball (ball_pt_pixel)
        pts_orig = np.array([
            [clicked_targets[0][0], clicked_targets[0][1]], # Target 1 (Class 1)
            [clicked_targets[1][0], clicked_targets[1][1]], # Target 2 (Class 2)
            [clicked_targets[2][0], clicked_targets[2][1]], # Target 3 (Class 3)
            [clicked_targets[3][0], clicked_targets[3][1]], # Target 4 (Class 4)
            [ball_pt_pixel[0], ball_pt_pixel[1]]            # Ball (Class 0)
        ], dtype=np.float32)
        
        # Generate random warp to simulate different camera angles
        H_warp = generate_random_warp(w, h)
        
        # Warp the image
        img_warped = cv2.warpPerspective(img, H_warp, (w, h), borderMode=cv2.BORDER_REPLICATE)
        
        # Warp the points
        pts_orig_reshaped = np.array([pts_orig]) # 1x5x2
        pts_warped = cv2.perspectiveTransform(pts_orig_reshaped, H_warp)[0]
        
        labels = []
        # Target Marker classes 1 to 4
        for i in range(4):
            mcx, mcy = pts_warped[i]
            mw, mh = 20.0, 20.0 # Marker bounding box size
            
            mcx_norm = max(0, min(1, mcx / w))
            mcy_norm = max(0, min(1, mcy / h))
            mw_norm = mw / w
            mh_norm = mh / h
            labels.append(f"{i+1} {mcx_norm:.6f} {mcy_norm:.6f} {mw_norm:.6f} {mh_norm:.6f}")
            
        # Ball class 0
        bcx, bcy = pts_warped[4]
        bw, bh = 30.0, 30.0 # Ball bounding box size
        bcx_norm = max(0, min(1, bcx / w))
        bcy_norm = max(0, min(1, bcy / h))
        bw_norm = bw / w
        bh_norm = bh / h
        labels.append(f"0 {bcx_norm:.6f} {bcy_norm:.6f} {bw_norm:.6f} {bh_norm:.6f}")
            
        # Save
        out_img_path = os.path.join(out_images_dir, img_name)
        out_txt_path = os.path.join(out_labels_dir, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))
        
        cv2.imwrite(out_img_path, img_warped)
        with open(out_txt_path, 'w') as f:
            f.write('\n'.join(labels) + '\n')
            
        success_count += 1
        if success_count % 100 == 0:
            print(f"Generated {success_count} warped labeled images...")
            
    print(f"Done! Successfully generated {success_count} images.")
    
    # create dataset.yaml
    yaml_content = f"""path: /content/ball_balance_video_controlled/host_software/ml_vision/data/05_warped_dataset
train: images
val: images

names:
  0: ball
  1: green_target
  2: red_target
  3: black_target
  4: grey_target
"""
    with open(os.path.join(output_dir, "dataset.yaml"), "w") as f:
        f.write(yaml_content)
    print("Created dataset.yaml for YOLO.")

if __name__ == '__main__':
    main()
