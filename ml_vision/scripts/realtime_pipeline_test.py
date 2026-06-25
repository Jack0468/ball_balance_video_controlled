"""
realtime_pipeline_test.py

An integrated test script that runs the entire ML Vision Pipeline:
Webcam Capture -> Edge Detection (Canny) -> Perspective Warp -> YOLO Inference.

Allows testing of head-to-head latency vs accuracy, with configurable 
fallbacks and headless execution for performance benchmarking.

use --headless to disable visual render
"""

import cv2
import time
import argparse
import os
import numpy as np
from ultralytics import YOLO

# Import our custom preprocessor
from preprocessor import Preprocessor

def run_pipeline(source=0, headless=False, fallback='raw', max_frames=None):
    print("Loading YOLOv8n pre-trained model...")
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "weights", "yolov8n.pt")
    model = YOLO(model_path)
    
    print("Initializing Canny Preprocessor...")
    preproc = Preprocessor(platform_size=(500, 500))
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return

    frame_count = 0
    total_latency = 0
    total_preproc_latency = 0
    total_infer_latency = 0
    
    print(f"\n--- Starting Realtime Pipeline ---")
    print(f"Headless mode: {headless}")
    print(f"Fallback mode: {fallback} (If Canny fails to find platform)")
    print("Press 'q' to quit at any time (if not headless).\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        start_time = time.time()
        
        # 1. PREPROCESSING (Edge Detection + Warp)
        preproc_start = time.time()
        M, warped_platform = preproc.get_perspective_transform(frame)
        preproc_time = (time.time() - preproc_start) * 1000
        
        # Determine which image to run inference on
        inference_frame = None
        warp_successful = False
        
        if M is not None and warped_platform is not None:
            inference_frame = warped_platform
            warp_successful = True
        else:
            if fallback == 'raw':
                inference_frame = frame
            elif fallback == 'skip':
                inference_frame = None
                
        # 2. YOLO INFERENCE
        infer_time = 0
        annotated_frame = None
        
        if inference_frame is not None:
            infer_start = time.time()
            results = model(inference_frame, classes=[32], verbose=False) # 32 = sports ball
            infer_time = (time.time() - infer_start) * 1000
            annotated_frame = results[0].plot()
            
            # Print coordinates if detected
            boxes = results[0].boxes
            if len(boxes) > 0:
                x1, y1, x2, y2 = boxes[0].xyxy[0].cpu().numpy()
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                status = "WARPED" if warp_successful else "RAW (FALLBACK)"
                # print(f"Frame {frame_count}: Ball at ({cx:.1f}, {cy:.1f}) on {status} image")
                
        latency = (time.time() - start_time) * 1000
        
        total_preproc_latency += preproc_time
        total_infer_latency += infer_time
        total_latency += latency
        frame_count += 1
        
        # Display statistics periodically in console
        if frame_count % 30 == 0:
            print(f"FPS: {1000/latency:.1f} | Preproc: {preproc_time:.1f}ms | YOLO: {infer_time:.1f}ms | Total: {latency:.1f}ms")

        # 3. GUI DISPLAY (If not headless)
        if not headless:
            if annotated_frame is not None:
                cv2.imshow("Pipeline Output (YOLO)", annotated_frame)
            cv2.imshow("Raw Camera Feed", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        if max_frames and frame_count >= max_frames:
            break

    cap.release()
    if not headless:
        cv2.destroyAllWindows()
        
    if frame_count > 0:
        print(f"\n--- Benchmark Results ({frame_count} frames) ---")
        print(f"Average Preprocessing Latency: {total_preproc_latency / frame_count:.1f}ms")
        print(f"Average YOLO Inference Latency: {total_infer_latency / frame_count:.1f}ms")
        print(f"Average Total Pipeline Latency: {total_latency / frame_count:.1f}ms")
        print(f"Average Pipeline FPS: {1000 / (total_latency / frame_count):.1f} FPS")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time ML Vision Pipeline Test")
    parser.add_argument("source", nargs="?", default=0, help="Webcam ID (default: 0) or video file path")
    parser.add_argument("--headless", action="store_true", help="Run without rendering cv2 GUI windows for pure latency testing")
    parser.add_argument("--fallback", choices=['raw', 'skip'], default='raw', 
                        help="Action to take if platform warping fails: run YOLO on 'raw' frame, or 'skip' inference entirely.")
    parser.add_argument("--max_frames", type=int, default=None, help="Max frames to process before stopping")
    
    args = parser.parse_args()
    
    source = int(args.source) if str(args.source).isdigit() else args.source
    run_pipeline(source, args.headless, args.fallback, args.max_frames)
