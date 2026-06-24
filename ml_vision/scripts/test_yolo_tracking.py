"""
test_yolo_tracking.py

Tests the pre-trained YOLOv8 model for tracking a sports ball.
Can run on a live webcam feed or a directory of images.
"""

import cv2
import time
import argparse
import os
from ultralytics import YOLO

def test_yolo(source, max_frames=None):
    print("Loading YOLOv8n pre-trained model...")
    model = YOLO("yolov8n.pt")
    
    # Check if source is an integer (webcam) or a directory/file
    try:
        source_id = int(source)
        cap = cv2.VideoCapture(source_id)
        is_video = True
        print(f"Opening webcam {source_id}")
    except ValueError:
        if os.path.isdir(source):
            print(f"Reading images from directory: {source}")
            is_video = False
            image_files = sorted([os.path.join(source, f) for f in os.listdir(source) if f.endswith(('.png', '.jpg', '.jpeg'))])
        else:
            print(f"Opening video file: {source}")
            is_video = True
            cap = cv2.VideoCapture(source)
            
    if is_video and not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return

    frame_count = 0
    total_latency = 0
    
    def process_frame(frame):
        nonlocal total_latency
        start_time = time.time()
        
        # Sports ball is class 32 in COCO
        results = model(frame, classes=[32], verbose=False)
        
        latency = (time.time() - start_time) * 1000
        total_latency += latency
        
        # Draw bounding boxes
        annotated_frame = results[0].plot()
        
        # Print center coordinates
        boxes = results[0].boxes
        if len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                print(f"Frame {frame_count}: Sports ball detected at ({cx:.1f}, {cy:.1f}) - Latency: {latency:.1f}ms")
        else:
            print(f"Frame {frame_count}: No ball detected - Latency: {latency:.1f}ms")
            
        return annotated_frame
        
    if is_video:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            out_frame = process_frame(frame)
            
            # cv2.imshow("YOLO Tracking", out_frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
                
            frame_count += 1
            if max_frames and frame_count >= max_frames:
                break
        cap.release()
    else:
        for img_path in image_files:
            frame = cv2.imread(img_path)
            if frame is None:
                continue
            out_frame = process_frame(frame)
            frame_count += 1
            if max_frames and frame_count >= max_frames:
                break
                
    if frame_count > 0:
        print(f"Average latency: {total_latency / frame_count:.1f}ms ({1000 / (total_latency / frame_count):.1f} FPS)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test pre-trained YOLO tracking.")
    parser.add_argument("source", help="Webcam ID (e.g. 0), video file path, or image directory")
    parser.add_argument("--max_frames", type=int, default=None, help="Max frames to process")
    args = parser.parse_args()
    
    test_yolo(args.source, args.max_frames)
