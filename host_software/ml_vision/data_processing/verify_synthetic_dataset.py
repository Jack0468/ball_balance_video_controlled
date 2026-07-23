import cv2
import os
import glob
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='host_software/ml_vision/data/03_synthetic_yolo')
    args = parser.parse_args()
    
    images_dir = os.path.join(args.dataset, 'images')
    labels_dir = os.path.join(args.dataset, 'labels')
    
    image_paths = glob.glob(os.path.join(images_dir, '*.jpg'))
    if not image_paths:
        print(f"No images found in {images_dir}")
        return
        
    print(f"Found {len(image_paths)} synthetic images. Press any key to advance, or ESC to exit.")
    
    cv2.namedWindow("Verify Synthetic Dataset", cv2.WINDOW_NORMAL)
    
    names = {
        0: "Platform", 1: "Ball", 2: "Blue", 3: "Grey", 4: "Black", 
        5: "Red", 6: "Green", 7: "Yellow", 8: "Cyan", 9: "Purple", 
        10: "Orange", 11: "Pink", 12: "Brown"
    }
    
    colors = {
        0: (0, 255, 0), 1: (0, 0, 255), 2: (255, 0, 0), 3: (128, 128, 128), 
        4: (0, 0, 0), 5: (0, 0, 255), 6: (0, 255, 0), 7: (0, 255, 255),
        8: (255, 255, 0), 9: (255, 0, 255), 10: (0, 165, 255), 
        11: (203, 192, 255), 12: (42, 42, 165)
    }
    
    for img_path in sorted(image_paths):
        img = cv2.imread(img_path)
        if img is None: continue
        
        img_h, img_w = img.shape[:2]
        
        basename = os.path.basename(img_path)
        label_path = os.path.join(labels_dir, basename.replace('.jpg', '.txt'))
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts: continue
                    
                    c_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])
                    
                    x1 = int((cx - w/2) * img_w)
                    y1 = int((cy - h/2) * img_h)
                    x2 = int((cx + w/2) * img_w)
                    y2 = int((cy + h/2) * img_h)
                    
                    color = colors.get(c_id, (255, 255, 255))
                    name = names.get(c_id, str(c_id))
                    
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, name, (x1, max(y1-5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
        cv2.imshow("Verify Synthetic Dataset", img)
        key = cv2.waitKey(0) & 0xFF
        if key == 27: # ESC
            break
            
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
