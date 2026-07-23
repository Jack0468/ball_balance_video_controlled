import cv2
import threading
import queue
import time
import struct
import numpy as np
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
import serial
import argparse

# --- Configuration ---
SERIAL_PORT = "COM3"
SERIAL_BAUD = 2000000 
MAX_BOUND = 200.0 # Denormalization constant
# ---------------------

class USBReceiver:
    def __init__(self, camera_id=0):
        self.cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(camera_id)
            
        # Force MJPG codec to prevent USB 2.0 bandwidth bottlenecks
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

def load_expert_model(model_path, device):
    print("Loading PyTorch ResNet18 Expert Model B...")
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
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
