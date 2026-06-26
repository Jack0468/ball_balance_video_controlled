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
    
    # NOTE: You will need to train the YOLO-Pose model on the synthetic dataset first!
    # Uncomment the line below once trained:
    # pose_model = YOLO('runs/pose/train/weights/best.pt')
    
    ball_model = YOLO('yolov8n.pt')
    preproc = Preprocessor()
    
    cap = cv2.VideoCapture(1) # Assuming camera index 1
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
        # STAGE 1: PLATFORM DETECTION (YOLO-Pose)
        # ==========================================
        warped = None
        M = None
        
        # --- UNCOMMENT WHEN POSE MODEL IS TRAINED ---
        # results_pose = pose_model(frame, verbose=False)
        # if len(results_pose[0].keypoints) > 0:
        #     # Extract 4 keypoints
        #     kps = results_pose[0].keypoints.xy[0].cpu().numpy()
        #     if len(kps) == 4:
        #         sorted_kps = sort_corners(kps)
        #         M = cv2.getPerspectiveTransform(sorted_kps, preproc.dst_pts)
        #         warped = cv2.warpPerspective(frame, M, preproc.platform_size)
        
        # Fallback to classical preprocessor if YOLO-Pose is not ready or fails
        if warped is None:
            M, warped = preproc.get_perspective_transform(frame)
            
        t_preproc = time.time()
        
        # ==========================================
        # STAGE 2: BALL DETECTION (YOLO-Ball)
        # ==========================================
        results_ball = ball_model(warped, classes=[32], verbose=False)
        t_yolo = time.time()
        
        # ==========================================
        # VISUALIZATION
        # ==========================================
        for r in results_ball:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                cv2.rectangle(warped, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
        cv2.imshow("Cascaded Output", warped)
        
        dt_preproc = (t_preproc - t0) * 1000
        dt_yolo = (t_yolo - t_preproc) * 1000
        dt_total = (t_yolo - t0) * 1000
        fps = 1000.0 / max(dt_total, 1.0)
        
        print(f"FPS: {fps:.1f} | Pose/Warp: {dt_preproc:.1f}ms | YOLO-Ball: {dt_yolo:.1f}ms")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
