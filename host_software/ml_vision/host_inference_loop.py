import cv2
import socket
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

# --- Configuration ---
UDP_IP = "0.0.0.0" # Listen on all interfaces
UDP_PORT = 5005
SERIAL_PORT = "COM3"
SERIAL_BAUD = 115200

# Target coordinate (can be changed dynamically later by multi-task heads)
TARGET_X = 0.0
TARGET_Y = 0.0
MAX_BOUND = 200.0 # Denormalization constant
# ---------------------

class UDPReceiver:
    def __init__(self, ip, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((ip, port))
        # Keep only the single most recent frame
        self.frame_queue = queue.Queue(maxsize=1)
        self.running = True
        self.thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.thread.start()
        
    def _receive_loop(self):
        print(f"UDP Receiver listening on {UDP_IP}:{UDP_PORT}")
        while self.running:
            try:
                # 65535 is the max UDP packet size, large enough for 640x480 quality 80 jpeg
                data, _ = self.sock.recvfrom(65535)
                # Decompress instantly
                frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                
                if frame is not None:
                    # If queue is full, drop the old frame to minimize latency
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                    self.frame_queue.put(frame)
            except Exception as e:
                if self.running:
                    print(f"UDP Error: {e}")

    def get_latest_frame(self):
        try:
            return self.frame_queue.get(timeout=1.0)
        except queue.Empty:
            return None

    def stop(self):
        self.running = False
        self.sock.close()

def load_expert_model(model_path, device):
    print("Loading PyTorch ResNet18 Expert Model...")
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    if os.path.exists(model_path):
        # We use weights_only=True for security against untrusted pickles, though not strictly required here
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded weights from {model_path}")
    else:
        print(f"WARNING: Weights {model_path} not found! Using random weights.")
        
    model = model.to(device)
    model.eval()
    return model

def main():
    # 1. Hardware/Model Init
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.abspath(os.path.join(script_dir, '../models/resnet18_expert_tracker/expert_tracker_best.pth'))
    
    model = load_expert_model(model_path, device)
    
    # Matching the transforms used in ball_dataset.py
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((240, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 2. Serial Port Init
    try:
        ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0)
        print(f"Connected to STM32 on {SERIAL_PORT} at {SERIAL_BAUD} baud.")
    except Exception as e:
        print(f"Could not open serial port {SERIAL_PORT}: {e}")
        print("Continuing in dry-run mode (no serial transmission).")
        ser = None

    # 3. UDP Stream Init
    receiver = UDPReceiver(UDP_IP, UDP_PORT)
    
    print("Starting Main Inference Loop...")
    try:
        while True:
            # Blocks until a new frame arrives
            frame = receiver.get_latest_frame()
            if frame is None:
                continue
                
            start_t = time.perf_counter()
            
            # --- Inference Phase ---
            # cv2 decodes as BGR, but our model was trained on RGB images loaded by PIL. 
            # We MUST convert the color space or the model will see swapped colors!
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = preprocess(rgb_frame).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(input_tensor)
            
            # Extract and denormalize coordinates
            norm_x, norm_y = output[0].cpu().numpy()
            touch_x = float(norm_x * MAX_BOUND)
            touch_y = float(norm_y * MAX_BOUND)
            
            # --- Error Calculation Phase ---
            error_x = TARGET_X - touch_x
            error_y = TARGET_Y - touch_y
            
            # --- Serial Transmission Phase ---
            # Format: [Start Byte '<'] [Error X (16-bit int)] [Error Y (16-bit int)] [End Byte '\n']
            # Using standard struct format: c=char, h=short (2 bytes signed)
            try:
                # Clamp errors to fit in 16-bit signed integer (-32768 to 32767)
                err_x_int = int(max(min(error_x, 32767), -32768))
                err_y_int = int(max(min(error_y, 32767), -32768))
                
                payload = struct.pack('<chh', b'<', err_x_int, err_y_int)
                if ser is not None:
                    ser.write(payload)
            except Exception as e:
                print(f"Failed to pack or send data: {e}")
                
            elapsed = (time.perf_counter() - start_t) * 1000.0
            print(f"Processed frame in {elapsed:.1f}ms | Error: X={err_x_int}, Y={err_y_int}")
            
    except KeyboardInterrupt:
        print("\nShutting down pipeline...")
    finally:
        receiver.stop()
        if ser is not None:
            ser.close()

if __name__ == '__main__':
    main()
