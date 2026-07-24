import cv2
import threading
import queue
import time
import struct
import numpy as np
import os
import torch
import torch.nn as nn
import serial
import argparse
from ml_vision.core.coordinate_math import HomographyProjector
from ml_audio.audio_receiver import AudioCommandReceiver

from src.receivers import USBReceiver, UDPReceiver
from src.utils import find_stm32_port
from src.models import load_yolo_model, load_corrector_model
from src.state_machine import TargetStateMachine

# --- Configuration ---
SERIAL_PORT = "COM3"
SERIAL_BAUD = 2000000 
# ---------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam_id", type=int, default=0, help="Camera ID for USB mode")
    parser.add_argument("--port", type=str, default="auto", help="STM32 serial port (e.g. COM3 or 'auto')")
    parser.add_argument("--udp", action="store_true", help="Use UDP receiver instead of USB camera")
    parser.add_argument("--udp_port", type=int, default=5001, help="Port to listen on for UDP video stream")
    args = parser.parse_args()

    # Auto-detect port if 'auto'
    if args.port == "auto":
        detected_port = find_stm32_port()
        if detected_port:
            print(f"Auto-detected STM32 on {detected_port}")
            args.port = detected_port
        else:
            args.port = SERIAL_PORT
            print(f"Could not auto-detect STM32. Defaulting to {args.port}")

    # 1. Hardware/Model Init
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    yolo_path = os.path.abspath(os.path.join(script_dir, 'ml_vision/models/platform_and_markers_model/weights/best.pt'))
    corrector_path = os.path.abspath(os.path.join(script_dir, 'ml_vision/models/corrector/best_corrector.pth'))
    
    yolo_model = load_yolo_model(yolo_path, device)
    corrector_model = load_corrector_model(corrector_path, device)
    
    # Initialize Homography Projector
    dst_pts = np.array([
        [-70, 55],
        [70, 55],
        [70, -55],
        [-70, -55]
    ], dtype=np.float32)
    projector = HomographyProjector(dst_pts)
    
    # 2. Audio & State Init
    audio_model_path = os.path.abspath(os.path.join(script_dir, 'ml_audio/models/audio_command_classifier/best_classifier.keras'))
    audio_receiver = AudioCommandReceiver(audio_model_path)
    state_machine = TargetStateMachine()
    
    # 3. Serial Port Init
    try:
        ser = serial.Serial(args.port, SERIAL_BAUD, timeout=0)
        print(f"Connected to STM32 on {args.port} at {SERIAL_BAUD} baud.")
    except Exception as e:
        print(f"Could not open serial port {args.port}. Continuing in dry-run mode (no serial transmission).")
        ser = None
        
    # 3. Camera/Receiver Init
    if args.udp:
        receiver = UDPReceiver(port=args.udp_port, width=640, height=480)
    else:
        receiver = USBReceiver(camera_id=args.cam_id)
        
    # Wait for the first frame
    print("Waiting for camera feed...")
    frame = None
    while frame is None:
        frame = receiver.get_latest_frame()
        time.sleep(0.1)

    print(f"Starting YOLO Main Inference Loop... (Headless mode, press Ctrl+C to quit)")
    try:
        while True:
            frame = receiver.get_latest_frame()
            if frame is None:
                continue
                
            start_t = time.perf_counter()
            
            # Inference Phase
            results = yolo_model.predict(source=frame, imgsz=320, conf=0.5, verbose=False)
            yolo_t = time.perf_counter()
            
            if not results or len(results) == 0:
                print("No YOLO results")
                continue
                
            res = results[0]
            if res.boxes is None:
                print("No boxes detected")
                continue
                
            classes = res.boxes.cls.cpu().numpy()
            boxes = res.boxes.xywh.cpu().numpy()
            
            corners = None
            ball_box = None
            detected_markers = {}
            
            for i, cls in enumerate(classes):
                c = int(cls)
                if c == 0: # Platform
                    if res.keypoints is not None and len(res.keypoints.xy) > i:
                        kpts = res.keypoints.xy[i].cpu().numpy()
                        if len(kpts) == 4:
                            corners = kpts
                elif c == 1: # Ball
                    ball_box = boxes[i]
                elif c >= 2: # Marker
                    name = yolo_model.names[c].replace('_marker', '')
                    detected_markers[name] = boxes[i]
            
            if corners is None or ball_box is None:
                end_t = time.perf_counter()
                fps = 1.0 / (end_t - start_t)
                yolo_latency_ms = (yolo_t - start_t) * 1000.0
                print(f"Missing detections - Platform: {'Found' if corners is not None else 'Missing'}, Ball: {'Found' if ball_box is not None else 'Missing'} | FPS: {fps:.1f} | YOLO Latency: {yolo_latency_ms:.1f}ms")
                continue
            
            homography_x, homography_y = 0.0, 0.0
            marker_coords = {}
            if projector.update_homography(corners):
                hx, hy = projector.project_point(ball_box[0], ball_box[1])
                if hx is not None and hy is not None:
                    homography_x, homography_y = hx, hy
                    
                for name, box in detected_markers.items():
                    mx, my = projector.project_point(box[0], box[1])
                    if mx is not None and my is not None:
                        marker_coords[name] = (mx, my)
                    
            features = np.array([
                ball_box[0], ball_box[1], ball_box[2], ball_box[3],
                corners[0][0], corners[0][1], corners[1][0], corners[1][1],
                corners[2][0], corners[2][1], corners[3][0], corners[3][1],
                homography_x, homography_y
            ], dtype=np.float32)
            
            features[0:12:2] /= 640.0
            features[1:12:2] /= 480.0
            features[12:] /= 100.0
            
            input_tensor = torch.tensor(features).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = corrector_model(input_tensor)
            
            mlp_t = time.perf_counter()
            
            cam_x, cam_y = output[0].cpu().numpy()
            
            # Process Audio Commands
            command = audio_receiver.get_latest_command()
            state_machine.process_command(command)
            target_x, target_y = state_machine.get_target_coords(marker_coords)
            
            # Serial Transmission Phase
            # We send cam_x, cam_y, target_x, target_y so the RL policy gets true absolute coords
            # and target coords, keeping tilt dynamics perfectly aligned.
            try:
                payload = f"{cam_x:.2f},{cam_y:.2f},{target_x:.2f},{target_y:.2f}\n".encode('ascii')
                if ser is not None:
                    ser.write(payload)
            except Exception as e:
                print(f"Serial Error: {e}")
                
            end_t = time.perf_counter()
            
            total_latency_ms = (end_t - start_t) * 1000.0
            yolo_latency_ms = (yolo_t - start_t) * 1000.0
            mlp_latency_ms = (mlp_t - yolo_t) * 1000.0
            fps = 1.0 / (end_t - start_t)
            
            marker_str = ", ".join([f"{name}=({x:.1f},{y:.1f})" for name, (x, y) in marker_coords.items()])
            marker_out = f" | Markers: {marker_str}" if marker_str else ""
            
            print(f"Targeting '{state_machine.current_target_name}' at X={target_x:.1f} Y={target_y:.1f} | Ball: X={cam_x:.1f} Y={cam_y:.1f} mm | FPS: {fps:.1f} {marker_out}")
                
    except KeyboardInterrupt:
        pass
    finally:
        receiver.stop()
        audio_receiver.stop()
        if ser:
            ser.close()
        cv2.destroyAllWindows()
        print("Inference loop stopped.")

if __name__ == '__main__':
    main()
