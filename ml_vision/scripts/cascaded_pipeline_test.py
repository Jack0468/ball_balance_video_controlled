import cv2
import time
import numpy as np
from ultralytics import YOLO
from preprocessor import Preprocessor

def sort_corners(pts):
    # Sorts points into: top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def main():
    print("Loading Models...")
    
    import os
    
    # Safely resolve the absolute paths so it doesn't matter what folder you run the script from
    bbox_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/platform_bbox_model-4/weights/best.pt'))
    
    # Load the trained YOLOv8 BBox model
    bbox_model = YOLO(bbox_model_path, task='detect')
    
    preproc = Preprocessor()
    
    cap = cv2.VideoCapture(0) # Assuming camera index 1
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
        
    print("--- Starting Cascaded Pipeline Test ---")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        t0 = time.time()
        
        # ==========================================
        # STAGE 1: PLATFORM DETECTION (YOLO BBox)
        # ==========================================
        warped = None
        M = None
        
        results = bbox_model(frame, verbose=False)
        t_pose_end = time.time()
        
        if len(results[0].boxes) > 0:
            # Get the highest confidence bounding box
            box = results[0].boxes[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            
            # Add padding to avoid edge clipping
            pad = 30
            h, w = frame.shape[:2]
            y1_pad = max(0, y1 - pad)
            y2_pad = min(h, y2 + pad)
            x1_pad = max(0, x1 - pad)
            x2_pad = min(w, x2 + pad)
            
            # Create a masked frame where everything outside the bounding box is black
            masked_frame = np.zeros_like(frame)
            masked_frame[y1_pad:y2_pad, x1_pad:x2_pad] = frame[y1_pad:y2_pad, x1_pad:x2_pad]
            
            # Draw the bounding box on the original frame for visualization
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "YOLO BBox", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # STAGE 2: CLASSICAL CV FALLBACK ON MASKED FRAME
            M, warped = preproc.get_perspective_transform(masked_frame)
        
        # Fallback to classical preprocessor if YOLO BBox is not ready or fails
        if warped is None:
            M, warped = preproc.get_perspective_transform(frame)
            
        t_warp_end = time.time()
        
        # ==========================================
        # VISUALIZATION
        # ==========================================
        cv2.imshow("Original Feed (Corners)", frame)
        if warped is not None:
            cv2.imshow("Cascaded Output (Top-Down)", warped)
        if hasattr(preproc, 'last_mask') and preproc.last_mask is not None:
            cv2.imshow("HSV Tuning Mask", preproc.last_mask)
        
        dt_bbox = (t_pose_end - t0) * 1000
        dt_warp = (t_warp_end - t_pose_end) * 1000
        dt_total = (t_warp_end - t0) * 1000
        fps = 1000.0 / max(dt_total, 1.0)
        
        print(f"FPS: {fps:.1f} | YOLO BBox: {dt_bbox:.1f}ms | Warp: {dt_warp:.1f}ms")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
