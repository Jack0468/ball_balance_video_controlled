"""
NOT USEFUL: Automated approach failed due to camera shake/background noise.
Use select_crop.py instead.
"""
import cv2
import numpy as np
import sys

def auto_detect_platform_via_ball(video_path, num_frames=200, padding=100):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Could not open video.")
        return None
        
    print(f"Reading {num_frames} frames to track the ball and determine platform bounding box...")
    
    ret, prev_frame = cap.read()
    if not ret:
        return None
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    ball_points = []
    
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Frame difference to find motion
        diff = cv2.absdiff(gray, prev_gray)
        _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
        
        # Find moving objects
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # The laptop screen flash is huge. The ball is small.
            if 50 < area < 8000: 
                x, y, w, h = cv2.boundingRect(cnt)
                # Ensure it's somewhat square (like a ball)
                aspect_ratio = w / float(h)
                if 0.3 < aspect_ratio < 3.0:
                    center_x = x + w // 2
                    center_y = y + h // 2
                    ball_points.append((center_x, center_y))
                    
        prev_gray = gray
        
    cap.release()
    
    if len(ball_points) < 10:
        print("Could not track the ball enough times to determine platform size.")
        return None
        
    # Get bounding box of all ball positions
    pts = np.array(ball_points)
    min_x, min_y = np.min(pts, axis=0)
    max_x, max_y = np.max(pts, axis=0)
    
    fh, fw = prev_gray.shape
    
    # Apply padding
    cx = max(0, min_x - padding)
    cy = max(0, min_y - padding)
    cw = min(fw - cx, (max_x - min_x) + 2 * padding)
    ch = min(fh - cy, (max_y - min_y) + 2 * padding)
    
    print(f"Tracked ball at {len(ball_points)} positions.")
    print(f"Calculated Platform Crop: x={cx}, y={cy}, w={cw}, h={ch}")
    return (cx, cy, cw, ch)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        video = sys.argv[1]
    else:
        video = 'host_software/ml_vision/data/01_bronze/video1/20260710_054604000_iOS.MOV'
    
    roi = auto_detect_platform_via_ball(video)
    print("ROI:", roi)
