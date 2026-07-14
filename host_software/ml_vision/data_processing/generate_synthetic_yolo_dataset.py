import os
import cv2
import numpy as np
import random
import glob
from pathlib import Path

def find_red_ball(image):
    """Find the red ball using HSV masking and return its YOLO bounding box (cx, cy, w, h)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Red has two ranges in HSV
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2
    
    # Optional morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
        
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    if area < 50: # too small
        return None
        
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Add a slight padding to the bounding box
    pad = 5
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(image.shape[1] - x, w + 2*pad)
    h = min(image.shape[0] - y, h + 2*pad)
    
    # Normalize
    img_h, img_w = image.shape[:2]
    cx = (x + w/2) / img_w
    cy = (y + h/2) / img_h
    nw = w / img_w
    nh = h / img_h
    
    return (cx, cy, nw, nh)

def draw_marker_with_glare(img, pt, color_bgr):
    """Draw a marker with alpha blending and simulated glare."""
    overlay = img.copy()
    
    # Base circle radius
    r = random.randint(12, 18)
    
    # Draw solid base circle on overlay
    cv2.circle(overlay, pt, r, color_bgr, -1)
    
    # Apply alpha blending for translucent look
    alpha = random.uniform(0.6, 0.9)
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    
    # Draw simulated glare (white-ish blob)
    if random.random() > 0.3: # 70% chance of glare
        glare_overlay = img.copy()
        glare_r = int(r * random.uniform(0.5, 1.2))
        glare_offset_x = random.randint(-int(r/2), int(r/2))
        glare_offset_y = random.randint(-int(r/2), int(r/2))
        glare_pt = (pt[0] + glare_offset_x, pt[1] + glare_offset_y)
        cv2.circle(glare_overlay, glare_pt, glare_r, (255, 255, 255), -1)
        
        # very soft blending for glare
        glare_alpha = random.uniform(0.4, 0.8)
        img = cv2.addWeighted(glare_overlay, glare_alpha, img, 1 - glare_alpha, 0)
        
        # blur the region slightly
        x1, y1 = max(0, pt[0]-r*2), max(0, pt[1]-r*2)
        x2, y2 = min(img.shape[1], pt[0]+r*2), min(img.shape[0], pt[1]+r*2)
        if x2 > x1 and y2 > y1:
            roi = img[y1:y2, x1:x2]
            img[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (5, 5), 0)
            
    return img

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.abspath(os.path.join(script_dir, "../data/02_silver/images"))
    output_dir = os.path.abspath(os.path.join(script_dir, "../data/03_synthetic_yolo"))
    
    out_images_dir = os.path.join(output_dir, "images")
    out_labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_labels_dir, exist_ok=True)
    
    image_paths = glob.glob(os.path.join(input_dir, "*.jpg")) + glob.glob(os.path.join(input_dir, "*.png"))
    if not image_paths:
        print(f"ERROR: No images found in {input_dir}")
        return
        
    random.shuffle(image_paths)
    
    # Marker colors in BGR
    # Classes: 1:Green, 2:Red, 3:Black, 4:Grey
    marker_colors = [
        (0, 200, 0),      # Green
        (0, 0, 200),      # Red
        (30, 30, 30),     # Black
        (128, 128, 128)   # Grey
    ]
    
    # For testing/Colab speed, generate 2000 images
    NUM_SYNTHETIC = 2000
    images_to_process = image_paths[:NUM_SYNTHETIC]
    print(f"Generating {min(NUM_SYNTHETIC, len(images_to_process))} synthetic images...")
    
    success_count = 0
    for idx, img_path in enumerate(images_to_process):
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        img_h, img_w = img.shape[:2]
        
        # 1. Find real ball bounding box
        ball_bbox = find_red_ball(img)
        labels = []
        if ball_bbox is not None:
            labels.append(f"0 {ball_bbox[0]:.6f} {ball_bbox[1]:.6f} {ball_bbox[2]:.6f} {ball_bbox[3]:.6f}")
            
        # 2. Pick 4 random points forming a perspective rectangle
        margin = 30
        min_w, max_w = 300, 800
        min_h, max_h = 300, 800
        
        cx, cy = random.randint(img_w//3, 2*img_w//3), random.randint(img_h//3, 2*img_h//3)
        w, h = random.randint(min_w, max_w), random.randint(min_h, max_h)
        
        pts = [
            [cx - w//2, cy - h//2], # TL
            [cx + w//2, cy - h//2], # TR
            [cx + w//2, cy + h//2], # BR
            [cx - w//2, cy + h//2]  # BL
        ]
        
        jitter_x = int(w * 0.15)
        jitter_y = int(h * 0.15)
        for pt in pts:
            pt[0] += random.randint(-jitter_x, jitter_x)
            pt[1] += random.randint(-jitter_y, jitter_y)
            pt[0] = max(margin, min(img_w - margin, pt[0]))
            pt[1] = max(margin, min(img_h - margin, pt[1]))
            
        # 3. Draw markers and create labels
        for i, pt in enumerate(pts):
            class_id = i + 1
            img = draw_marker_with_glare(img, tuple(pt), marker_colors[i])
            mcx = pt[0] / img_w
            mcy = pt[1] / img_h
            mw = random.uniform(30, 45) / img_w
            mh = random.uniform(30, 45) / img_h
            labels.append(f"{class_id} {mcx:.6f} {mcy:.6f} {mw:.6f} {mh:.6f}")
            
        # 4. Save
        basename = os.path.basename(img_path)
        out_name = f"synth_{idx:05d}_{basename}"
        
        cv2.imwrite(os.path.join(out_images_dir, out_name), img)
        with open(os.path.join(out_labels_dir, out_name.replace(".jpg", ".txt").replace(".png", ".txt")), "w") as f:
            f.write("\n".join(labels) + "\n")
            
        success_count += 1
        if success_count % 100 == 0:
            print(f"Generated {success_count} / {NUM_SYNTHETIC}")
            
    print(f"Finished! Successfully generated {success_count} synthetic images.")
    
    # create dataset.yaml for YOLO
    yaml_content = f"""path: /content/ball_balance_video_controlled/host_software/ml_vision/data/03_synthetic_yolo
train: images
val: images

names:
  0: ball
  1: green_marker
  2: red_marker
  3: black_marker
  4: grey_marker
"""
    with open(os.path.join(output_dir, "dataset.yaml"), "w") as f:
        f.write(yaml_content)
    print("Created dataset.yaml for YOLO.")

if __name__ == "__main__":
    main()
