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
import csv

# --- Configuration ---
SERIAL_PORT = "COM7"
SERIAL_BAUD = 2000000 # Increased to 2Mbaud to match firmware
MAX_BOUND = 200.0 # Denormalization constant
LOG_FILE = "laptop_camera_telemetry.csv"
# ---------------------

class TelemetryLogger:
    def __init__(self, serial_port):
        self.ser = serial_port
        self.running = True
        self.log_queue = queue.Queue()
        
        # Open CSV file and write header
        self.csv_file = open(LOG_FILE, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "host_time_ms", "mcu_micros", "target_x", "target_y", 
            "touch_x", "touch_y", "cam_x", "cam_y", "error_x", "error_y",
            "pitch", "roll", "theta_a", "theta_b", "theta_c",
            "integral_x", "integral_y", "deriv_x", "deriv_y"
        ])
        
        # Telemetry struct format: 2 uint32, 15 floats = 68 bytes
        self.struct_fmt = '<IIfffffffffffffff'
        self.struct_len = struct.calcsize(self.struct_fmt)
        self.sync_word = b'\xAA\xBB\xCC\xDD'
        
        # Store the latest camera predictions so we can inject them into the log
        self.latest_cam_x = 0.0
        self.latest_cam_y = 0.0
        
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        if self.ser is not None:
            self.thread.start()
            
    def _read_loop(self):
        buffer = bytearray()
        while self.running:
            try:
                if self.ser.in_waiting > 0:
                    buffer.extend(self.ser.read(self.ser.in_waiting))
                    
                    while len(buffer) >= self.struct_len:
                        # Find sync word
                        sync_idx = buffer.find(self.sync_word)
                        if sync_idx == -1:
                            buffer.clear()
                            break
                            
                        if sync_idx + self.struct_len <= len(buffer):
                            packet = buffer[sync_idx : sync_idx + self.struct_len]
                            self._parse_packet(packet)
                            buffer = buffer[sync_idx + self.struct_len :]
                        else:
                            # Wait for more data
                            buffer = buffer[sync_idx:]
                            break
                else:
                    time.sleep(0.001)
            except Exception as e:
                if self.running:
                    print(f"Telemetry Read Error: {e}")
                    time.sleep(0.1)

    def _parse_packet(self, packet):
        try:
            unpacked = struct.unpack(self.struct_fmt, packet)
            # unpacked[0] is sync_header
            mcu_micros = unpacked[1]
            target_x = unpacked[2]
            target_y = unpacked[3]
            touch_x = unpacked[4]
            touch_y = unpacked[5]
            error_x = unpacked[6]
            error_y = unpacked[7]
            pitch = unpacked[8]
            roll = unpacked[9]
            theta_a = unpacked[10]
            theta_b = unpacked[11]
            theta_c = unpacked[12]
            integral_x = unpacked[13]
            integral_y = unpacked[14]
            deriv_x = unpacked[15]
            deriv_y = unpacked[16]
            
            host_time = int(time.time() * 1000)
            
            self.csv_writer.writerow([
                host_time, mcu_micros, target_x, target_y, 
                touch_x, touch_y, self.latest_cam_x, self.latest_cam_y,
                error_x, error_y, pitch, roll, 
                theta_a, theta_b, theta_c,
                integral_x, integral_y, deriv_x, deriv_y
            ])
        except struct.error:
            pass

    def update_cam_pos(self, x, y):
        self.latest_cam_x = x
        self.latest_cam_y = y

    def stop(self):
        self.running = False
        if self.ser is not None:
            self.thread.join(timeout=1.0)
        self.csv_file.close()

class USBReceiver:
    def __init__(self, camera_id=0):
        self.cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(camera_id)
            
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

def load_expert_model(model_path, device):
    print("Loading PyTorch ResNet18 Expert Model (Subset)...")
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

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam_id", type=int, default=0, help="Camera ID for USB mode")
    args = parser.parse_args()

    # 1. Hardware/Model Init
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.abspath(os.path.join(script_dir, 'models/resnet18_expert_tracker_subset/expert_tracker_subset_best.pth'))
    
    model = load_expert_model(model_path, device)
    
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

    # 3. Stream & Logger Init
    receiver = USBReceiver(args.cam_id)
    logger = TelemetryLogger(ser)
    
    print(f"Starting Main Inference Loop...")
    try:
        while True:
            frame = receiver.get_latest_frame()
            if frame is None:
                continue
                
            start_t = time.perf_counter()
            
            # Inference Phase
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = preprocess(rgb_frame).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(input_tensor)
            
            norm_x, norm_y = output[0].cpu().numpy()
            cam_x = float(norm_x * MAX_BOUND)
            cam_y = float(norm_y * MAX_BOUND)
            
            # Update logger
            logger.update_cam_pos(cam_x, cam_y)
            
            # Serial Transmission Phase
            try:
                cam_x_int = int(max(min(cam_x, 32767), -32768))
                cam_y_int = int(max(min(cam_y, 32767), -32768))
                
                payload = struct.pack('<chh', b'<', cam_x_int, cam_y_int)
                if ser is not None:
                    ser.write(payload)
            except Exception as e:
                print(f"Serial Error: {e}")
                
            end_t = time.perf_counter()
            fps = 1.0 / (end_t - start_t)
            
            # Visualization Phase
            cv2.putText(frame, f"Cam: X={cam_x:.1f} Y={cam_y:.1f} mm", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 200, 0), 2)
                        
            cv2.imshow("Live Inference (Press 'q' to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        receiver.stop()
        logger.stop()
        if ser:
            ser.close()
        cv2.destroyAllWindows()
        print("Inference loop stopped. Telemetry saved to laptop_camera_telemetry.csv.")

if __name__ == '__main__':
    main()
