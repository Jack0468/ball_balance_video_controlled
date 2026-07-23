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
from ultralytics import YOLO
from ml_vision.core.coordinate_math import HomographyProjector
from ml_vision.core.corrector_mlp import CorrectorMLP

# --- Configuration ---
SERIAL_PORT = "COM3"
SERIAL_BAUD = 2000000 
# ---------------------

class USBReceiver:
    def __init__(self, camera_id=0):
        self.cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(camera_id)
            
        # Force MJPG codec to prevent USB 2.0 bandwidth bottlenecks from capping FPS at 10-15!
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
            
        self.frame_queue = queue.Queue(maxsize=1)
        self.running = True
        self.thread = threading.Thread(target=self._receive_loop, daemon=True)
        if self.cap.isOpened():
            self.thread.start()
            print(f"USB Camera {camera_id} initialized.")
        else:
            print(f"ERROR: Could not open USB Camera {camera_id}")
            
    def _receive_loop(self):
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                if frame.shape[:2] != (480, 640):
                    frame = cv2.resize(frame, (640, 480))
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put(frame)
            else:
                time.sleep(0.01)

    def get_latest_frame(self):
        try:
            return self.frame_queue.get(timeout=1.0)
        except queue.Empty:
            return None

    def stop(self):
        self.running = False
        self.cap.release()

import socket

class UDPReceiver:
    def __init__(self, port=5001, width=640, height=480):
        self.port = port
        self.width = width
        self.height = height
        self.pixel_bytes = 2
        self.frame_size = self.width * self.height * self.pixel_bytes
        self.packet_payload = 1024
        self.packets_per_frame = (self.frame_size + self.packet_payload - 1) // self.packet_payload
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('0.0.0.0', self.port))
        self.sock.settimeout(1.0)
        
        self.frame_queue = queue.Queue(maxsize=1)
        self.running = True
        self.thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.thread.start()
        print(f"UDP Receiver initialized on port {port}.")

    def _receive_loop(self):
        frame_buffer = bytearray(self.frame_size)
        current_frame_id = -1
        packets_received = 0
        
        while self.running:
            try:
                data, addr = self.sock.recvfrom(2048)
                if len(data) < 4:
                    continue
                    
                frame_id = struct.unpack('<H', data[0:2])[0]
                packet_id = struct.unpack('<H', data[2:4])[0]
                payload = data[4:]
                
                if frame_id != current_frame_id:
                    if current_frame_id != -1 and packets_received > self.packets_per_frame * 0.8:
                        # Process previous frame
                        img_np = np.frombuffer(frame_buffer, dtype=np.uint16).reshape((self.height, self.width))
                        # Convert RGB565 to BGR
                        b = ((img_np & 0x001F) << 3).astype(np.uint8)
                        g = ((img_np & 0x07E0) >> 3).astype(np.uint8)
                        r = ((img_np & 0xF800) >> 8).astype(np.uint8)
                        bgr_frame = cv2.merge([b, g, r])
                        
                        if self.frame_queue.full():
                            try:
                                self.frame_queue.get_nowait()
                            except queue.Empty:
                                pass
                        self.frame_queue.put(bgr_frame)
                        
                    current_frame_id = frame_id
                    packets_received = 0
                    
                offset = packet_id * self.packet_payload
                length = len(payload)
                if offset + length <= self.frame_size:
                    frame_buffer[offset:offset+length] = payload
                    packets_received += 1
                    
            except socket.timeout:
                pass
            except Exception as e:
                print(f"UDP Error: {e}")
                time.sleep(0.01)

    def get_latest_frame(self):
        try:
            return self.frame_queue.get(timeout=1.0)
        except queue.Empty:
            return None

    def stop(self):
        self.running = False
        self.sock.close()

def load_yolo_model(model_path, device):
    print(f"Loading YOLO model from {model_path}...")
    model = YOLO(model_path)
    model.to(device)
    return model

def load_corrector_model(model_path, device):
    print(f"Loading CorrectorMLP from {model_path}...")
    model = CorrectorMLP()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded weights from {model_path}")
    else:
        print(f"WARNING: Weights {model_path} not found! Using random weights.")
        
    model = model.to(device)
    model.eval()
    return model

def find_stm32_port():
    import serial.tools.list_ports
    ports = serial.tools.list_ports.comports()
    if len(ports) == 1:
        return ports[0].device
    for p in ports:
        if "STMicroelectronics" in p.description or "STM" in p.description or "USB Serial" in p.description:
            return p.device
    return None

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
    yolo_path = os.path.abspath(os.path.join(script_dir, 'ml_vision/models/new_platform_pose_model/weights/best.pt'))
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

    print(f"Starting YOLO Main Inference Loop... (Headless mode, press Ctrl+C to quit)")
    try:
        while True:
            frame = receiver.get_latest_frame()
            if frame is None:
                continue
                
            start_t = time.perf_counter()
            
            # Inference Phase
            results = yolo_model.predict(source=frame, imgsz=640, conf=0.5, verbose=False)
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
            
            for i, cls in enumerate(classes):
                if int(cls) == 0: # Platform
                    if res.keypoints is not None and len(res.keypoints.xy) > i:
                        kpts = res.keypoints.xy[i].cpu().numpy()
                        if len(kpts) == 4:
                            corners = kpts
                elif int(cls) == 1: # Ball
                    ball_box = boxes[i]
            
            if corners is None or ball_box is None:
                end_t = time.perf_counter()
                fps = 1.0 / (end_t - start_t)
                yolo_latency_ms = (yolo_t - start_t) * 1000.0
                print(f"Missing detections - Platform: {'Found' if corners is not None else 'Missing'}, Ball: {'Found' if ball_box is not None else 'Missing'} | FPS: {fps:.1f} | YOLO Latency: {yolo_latency_ms:.1f}ms")
                continue
            
            homography_x, homography_y = 0.0, 0.0
            if projector.update_homography(corners):
                hx, hy = projector.project_point(ball_box[0], ball_box[1])
                if hx is not None and hy is not None:
                    homography_x, homography_y = hx, hy
                    
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
            
            # Serial Transmission Phase
            try:
                payload = f"{-cam_x:.2f},{-cam_y:.2f}\n".encode('ascii')
                if ser is not None:
                    ser.write(payload)
            except Exception as e:
                print(f"Serial Error: {e}")
                
            end_t = time.perf_counter()
            
            total_latency_ms = (end_t - start_t) * 1000.0
            yolo_latency_ms = (yolo_t - start_t) * 1000.0
            mlp_latency_ms = (mlp_t - yolo_t) * 1000.0
            fps = 1.0 / (end_t - start_t)
            
            print(f"Target: X={cam_x:.1f} Y={cam_y:.1f} mm | FPS: {fps:.1f} | Latency: Total={total_latency_ms:.1f}ms (YOLO={yolo_latency_ms:.1f}ms, MLP+Rest={mlp_latency_ms:.1f}ms)")
                
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
