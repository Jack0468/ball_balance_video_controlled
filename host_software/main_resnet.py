import cv2
import threading
import queue
import time
import struct
import numpy as np
import os
import torch
import argparse

from src.receivers import USBReceiver, UDPReceiver
from src.utils import find_stm32_port
from src.models import load_expert_model

# --- Configuration ---
SERIAL_PORT = "COM3"
SERIAL_BAUD = 2000000 
MAX_BOUND = 200.0 # Denormalization constant
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
    model_path = os.path.abspath(os.path.join(script_dir, 'ml_vision/models/resnet18_expert_tracker_B/expert_tracker_best.pth'))
    
    model = load_expert_model(model_path, device)
    
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((240, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 2. Serial Port Init
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

    print("\n--- ROI Selection ---")
    print("1. Click and drag to draw a bounding box around the platform.")
    print("2. Press SPACE or ENTER to confirm your selection.")
    print("---------------------\n")
    
    # Wait for the first frame
    frame = None
    for _ in range(30):
        frame = receiver.get_latest_frame()
        if frame is not None:
            break
        time.sleep(0.1)
        
    if frame is not None:
        roi = cv2.selectROI("Select Platform Bounds", frame, showCrosshair=True, fromCenter=False)
        cv2.destroyAllWindows()
    else:
        roi = (0, 0, 0, 0)
        
    if roi == (0, 0, 0, 0):
        print("No ROI selected, using full frame.")
        roi = None
    else:
        print(f"Selected ROI: {roi}")
    
    print(f"Starting Main Inference Loop... (Headless mode, press Ctrl+C to quit)")
    try:
        while True:
            frame = receiver.get_latest_frame()
            if frame is None:
                continue
                
            start_t = time.perf_counter()
            
            # Crop frame if ROI is selected
            if roi is not None:
                x, y, w, h = roi
                crop_frame = frame[y:y+h, x:x+w]
            else:
                crop_frame = frame.copy()
            
            # Inference Phase
            rgb_frame = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2RGB)
            input_tensor = preprocess(rgb_frame).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(input_tensor)
            
            resnet_t = time.perf_counter()
            
            norm_x, norm_y = output[0].cpu().numpy()
            cam_x = float(norm_x * MAX_BOUND)
            cam_y = float(norm_y * MAX_BOUND)
            
            # Serial Transmission Phase
            try:
                payload = f"{-cam_x:.2f},{-cam_y:.2f}\n".encode('ascii')
                if ser is not None:
                    ser.write(payload)
            except Exception as e:
                print(f"Serial Error: {e}")
                
            end_t = time.perf_counter()
            
            total_latency_ms = (end_t - start_t) * 1000.0
            resnet_latency_ms = (resnet_t - start_t) * 1000.0
            rest_latency_ms = (end_t - resnet_t) * 1000.0
            fps = 1.0 / (end_t - start_t)
            
            print(f"Target: X={cam_x:.1f} Y={cam_y:.1f} mm | FPS: {fps:.1f} | Latency: Total={total_latency_ms:.1f}ms (ResNet={resnet_latency_ms:.1f}ms, Comm={rest_latency_ms:.1f}ms)")
                
    except KeyboardInterrupt:
        pass
    finally:
        receiver.stop()
        if ser:
            ser.close()
        cv2.destroyAllWindows()
        print("Inference loop stopped.")

if __name__ == '__main__':
    main()
