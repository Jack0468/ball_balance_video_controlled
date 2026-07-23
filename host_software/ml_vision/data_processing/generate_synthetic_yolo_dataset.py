import os
import cv2
import numpy as np
import random
import glob

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

def random_point_in_quad(pts, existing_points=None, min_dist=45):
    # pts: numpy array of 4 points [TL, TR, BR, BL]
    if existing_points is None:
        existing_points = []
        
    for _ in range(100): # max 100 attempts to find a non-overlapping point
        u = random.uniform(0.1, 0.9)
        v = random.uniform(0.1, 0.9)
        
        top = (1 - u) * pts[0] + u * pts[1]
        bottom = (1 - u) * pts[3] + u * pts[2]
        p = (1 - v) * top + v * bottom
        pt = (int(p[0]), int(p[1]))
        
        # Check distance to existing points
        valid = True
        for ep in existing_points:
            dist = np.sqrt((pt[0] - ep[0])**2 + (pt[1] - ep[1])**2)
            if dist < min_dist:
                valid = False
                break
                
        if valid:
            return pt
            
    return None # Return None if no valid spot found

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.abspath(os.path.join(script_dir, "../data/yolo_raw_dataset"))
    output_dir = os.path.abspath(os.path.join(script_dir, "../data/03_synthetic_yolo"))
    
    in_images_dir = os.path.join(input_dir, "images")
    in_labels_dir = os.path.join(input_dir, "labels")
    
    out_images_dir = os.path.join(output_dir, "images")
    out_labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_labels_dir, exist_ok=True)
    
    # Clear old generated files
    for f in glob.glob(os.path.join(out_images_dir, "*")):
        os.remove(f)
    for f in glob.glob(os.path.join(out_labels_dir, "*")):
        os.remove(f)
        
    image_paths = glob.glob(os.path.join(in_images_dir, "*.jpg"))
    if not image_paths:
        print(f"ERROR: No images found in {in_images_dir}")
        return
        
    # Marker classes and colors in BGR
    markers_config = [
        (2, (255, 0, 0)),      # Blue
        (3, (128, 128, 128)),  # Grey
        (4, (30, 30, 30)),     # Black
        (5, (0, 0, 200)),      # Red
        (6, (0, 200, 0)),      # Green
        (7, (0, 255, 255)),    # Yellow
        (8, (255, 255, 0)),    # Cyan
        (9, (255, 0, 255)),    # Purple
        (10, (0, 165, 255)),   # Orange
        (11, (203, 192, 255)), # Pink
        (12, (42, 42, 165))    # Brown
    ]
    
    augmentations_per_image = 20
    print(f"Generating {augmentations_per_image} synthetic variations for {len(image_paths)} images...")
    
    success_count = 0
    for img_path in image_paths:
        basename = os.path.basename(img_path)
        label_path = os.path.join(in_labels_dir, basename.replace(".jpg", ".txt"))
        
        if not os.path.exists(label_path):
            continue
            
        # Parse ground truth
        platform_keypoints = None
        base_labels = []
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts: continue
                c_id = int(parts[0])
                # Retain all original labels, including the real markers that the user manually labeled!
                base_labels.append(line.strip())
                
                # If platform, extract keypoints
                if c_id == 0 and len(parts) >= 17:
                    # keypoints start at index 5: x1 y1 v1 x2 y2 v2 ...
                    platform_keypoints = []
                    for i in range(4):
                        kx = float(parts[5 + i*3])
                        ky = float(parts[6 + i*3])
                        platform_keypoints.append([kx, ky])
        
        if not platform_keypoints:
            continue # Skip images without platform keypoints
            
        base_img = cv2.imread(img_path)
        if base_img is None:
            continue
            
        img_h, img_w = base_img.shape[:2]
        pts = np.array(platform_keypoints)
        pts[:, 0] *= img_w
        pts[:, 1] *= img_h
        
        for aug_idx in range(augmentations_per_image):
            img = base_img.copy()
            labels = list(base_labels) # Start with base labels (platform + ball)
            
            # Keep track of existing points to avoid overlap
            current_points = []
            for lbl in base_labels:
                parts = lbl.strip().split()
                if not parts: continue
                # if class is not platform (0), track its center
                if int(parts[0]) != 0:
                    cx = float(parts[1]) * img_w
                    cy = float(parts[2]) * img_h
                    current_points.append((cx, cy))
            
            # Place each marker
            for class_id, color_bgr in markers_config:
                pt = random_point_in_quad(pts, current_points, min_dist=45)
                if pt is None:
                    print(f"Warning: Could not find valid spot for class {class_id} in {basename}")
                    continue
                    
                current_points.append(pt)
                img = draw_marker_with_glare(img, pt, color_bgr)
                
                # Create label (assume 20x20 bounding box roughly, relative to img size)
                mcx = pt[0] / img_w
                mcy = pt[1] / img_h
                mw = random.uniform(30, 45) / img_w
                mh = random.uniform(30, 45) / img_h
                
                # YOLO format: class_id cx cy w h (and pad 0s for keypoints if pose model)
                m_label = f"{class_id} {mcx:.6f} {mcy:.6f} {mw:.6f} {mh:.6f}"
                for _ in range(4):
                    m_label += " 0.000000 0.000000 0"
                labels.append(m_label)
                
            out_name = f"synth_{aug_idx:03d}_{basename}"
            cv2.imwrite(os.path.join(out_images_dir, out_name), img)
            with open(os.path.join(out_labels_dir, out_name.replace(".jpg", ".txt")), "w") as f:
                f.write("\n".join(labels) + "\n")
                
            success_count += 1
            
        if success_count % 100 == 0:
            print(f"Generated {success_count} synthetic images...")
            
    print(f"Finished! Successfully generated {success_count} synthetic images.")

if __name__ == "__main__":
    main()
