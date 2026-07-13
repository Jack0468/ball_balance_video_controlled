import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import cv2
import random
import numpy as np
import albumentations as A

def load_dataset(input_dir):
    labels_file = os.path.join(input_dir, "labels.txt")
    dataset = []
    
    with open(labels_file, "r") as f:
        lines = f.readlines()[1:] # Skip header
        
    for line in lines:
        parts = line.strip().split(",")
        filename = parts[0]
        # Coordinates: x1,y1,x2,y2,x3,y3,x4,y4
        # Format for albumentations keypoints: [(x1,y1), (x2,y2), ...]
        coords = [float(x) for x in parts[1:]]
        keypoints = [(coords[i], coords[i+1]) for i in range(0, 8, 2)]
        
        img_path = os.path.join(input_dir, filename)
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                dataset.append((img, keypoints, filename))
        else:
            print(f"Image {img_path} not found.")
            
    return dataset

def get_augmentations():
    return A.Compose([
        A.Perspective(scale=(0.05, 0.2), p=0.8),
        A.Affine(translate_percent=(-0.1, 0.1), scale=(0.8, 1.2), rotate=(-30, 30), p=0.8),
        A.RandomBrightnessContrast(p=0.8),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.8)
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

def generate_yolo_labels(keypoints, img_width, img_height):
    # YOLO Pose format: class x_center y_center width height px1 py1 ...
    xs = [kp[0] for kp in keypoints]
    ys = [kp[1] for kp in keypoints]
    
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    
    # Clip to image boundaries
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(img_width, xmax)
    ymax = min(img_height, ymax)
    
    # Calculate YOLO bbox
    w = xmax - xmin
    h = ymax - ymin
    cx = xmin + w / 2
    cy = ymin + h / 2
    
    # Normalize
    cx /= img_width
    cy /= img_height
    w /= img_width
    h /= img_height
    
    # Class is 0 (platform)
    label_str = f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
    return label_str

def generate_dataset(dataset, output_dir, total_images=1000):
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    transform = get_augmentations()
    
    for i in range(total_images):
        img, keypoints, orig_filename = random.choice(dataset)
        
        try:
            transformed = transform(image=img, keypoints=keypoints)
            t_img = transformed['image']
            t_kps = transformed['keypoints']
            
            if len(t_kps) != 4:
                continue
                
            img_h, img_w, _ = t_img.shape
            
            out_filename = f"aug_{i:05d}_{orig_filename}"
            out_img_path = os.path.join(images_dir, out_filename)
            t_img_bgr = cv2.cvtColor(t_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(out_img_path, t_img_bgr)
            
            out_label_filename = out_filename.replace(".jpg", ".txt")
            out_label_path = os.path.join(labels_dir, out_label_filename)
            
            label_str = generate_yolo_labels(t_kps, img_w, img_h)
            with open(out_label_path, "w") as f:
                f.write(label_str + "\n")
                
            if i % 100 == 0:
                print(f"Generated {i}/{total_images} images...")
                
        except Exception as e:
            print(f"Error during augmentation: {e}")
            continue
            
    print(f"Finished generating {total_images} augmented images in {output_dir}")

if __name__ == "__main__":
    # Assuming script is run from the ml_vision/scripts/ directory
    input_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/02_silver/base_images"))
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/03_gold/augmented_dataset"))
    
    print(f"Loading dataset from {input_dir}")
    dataset = load_dataset(input_dir)
    
    if len(dataset) > 0:
        print(f"Loaded {len(dataset)} base images. Starting augmentation...")
        generate_dataset(dataset, output_dir, total_images=1000)
    else:
        print("No base images found. Did you run convert_heic.py?")
