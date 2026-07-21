import cv2
import numpy as np
import time
import serial
import struct
import threading
import queue
import socket
import os
import sys
from ultralytics import YOLO

SERIAL_PORT = 'COM3'
SERIAL_BAUD = 115200

class UDPReceiver:
    def __init__(self, port=8080):
        self.port = port
        self.width = 640
        self.height = 480
        self.pixel_bytes = 2
        self.frame_size = self.width * self.height * self.pixel_bytes
        self.packet_payload = 1024
        self.packets_per_frame = (self.frame_size + self.packet_payload - 1) // self.packet_payload
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024 * 5)
        self.sock.bind(("0.0.0.0", self.port))
            
        self.frame_queue = queue.Queue(maxsize=1)
        self.running = True
        self.thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.thread.start()
        print(f"UDP Receiver initialized on port {self.port}.")
            
    def _receive_loop(self):
        current_frame_id = -1
        frame_buffer = bytearray(self.frame_size)
        packets_received = 0
        
        while self.running:
            try:
                # Use a timeout so we can exit cleanly
                self.sock.settimeout(1.0)
                try:
                    data, addr = self.sock.recvfrom(2048)
                except socket.timeout:
                    continue
                
                if len(data) < 8:
                    continue
                    
                frame_id, packet_id = struct.unpack("<II", data[:8])
                payload = data[8:]
                
                if frame_id != current_frame_id:
                    if current_frame_id != -1 and packets_received > self.packets_per_frame * 0.8:
                        self._process_and_queue_frame(frame_buffer)
                    
                    current_frame_id = frame_id
                    packets_received = 0
                    
                offset = packet_id * self.packet_payload
                length = len(payload)
                
                if offset + length <= self.frame_size:
                    frame_buffer[offset:offset+length] = payload
                    packets_received += 1
                    
            except Exception as e:
                if self.running:
                    print(f"UDP Rx Error: {e}")
                    
    def _process_and_queue_frame(self, frame_bytes):
        try:
            img16 = np.frombuffer(frame_bytes, dtype=np.uint16).reshape((self.height, self.width))
            r = (img16 >> 11) & 0x1F
            g = (img16 >> 5) & 0x3F
            b = img16 & 0x1F
            
            r = (r * 255) // 31
            g = (g * 255) // 63
            b = (b * 255) // 31
            
            img_bgr = np.dstack((b, g, r)).astype(np.uint8)
            
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put(img_bgr)
        except Exception as e:
            pass

    def get_latest_frame(self):
        try:
            return self.frame_queue.get(timeout=1.0)
        except queue.Empty:
            return None

    def stop(self):
        self.running = False
        self.sock.close()

def get_platform_corners_and_ball(results):
    corners = None
    ball_center = None
    
    if not results or len(results) == 0:
        return None, None
        
    res = results[0]
    if res.boxes is None:
        return None, None
        
    classes = res.boxes.cls.cpu().numpy()
    boxes = res.boxes.xywh.cpu().numpy()
    
    for i, cls in enumerate(classes):
        if int(cls) == 0: # Platform
            if res.keypoints is not None and len(res.keypoints.xy) > i:
                kpts = res.keypoints.xy[i].cpu().numpy()
                if len(kpts) == 4:
                    corners = kpts
        elif int(cls) == 1: # Ball
            bx, by, bw, bh = boxes[i]
            ball_center = (bx, by)
            
    return corners, ball_center

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam_id", type=int, default=0, help="Camera ID for USB mode")
    args = parser.parse_args()

    # 1. Model Init
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.abspath(os.path.join(script_dir, 'models/new_platform_pose_model/weights/best.pt'))
    
    print(f"Loading YOLOv8-Pose Model from {model_path}...")
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}. You must run train_platform_pose.py first!")
        sys.exit(1)
        
    model = YOLO(model_path)
    
    # 2. Serial Port Init
    try:
        ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0)
        print(f"Connected to STM32 on {SERIAL_PORT} at {SERIAL_BAUD} baud.")
    except Exception as e:
        print(f"Could not open serial port {SERIAL_PORT}: {e}")
        print("Continuing in dry-run mode (no serial transmission).")
        ser = None

    # 3. Stream Init
    receiver = UDPReceiver(port=8080)
    
    # Physical dimensions of the board in mm
    # Coordinate system: (0,0) is center.
    # Top-Left: (-70, 55)
    # Top-Right: (70, 55)
    # Bottom-Right: (70, -55)
    # Bottom-Left: (-70, -55)
    dst_pts = np.array([
        [-70, 55],
        [70, 55],
        [70, -55],
        [-70, -55]
    ], dtype=np.float32)
    
    print(f"Starting Main Inference Cascade Loop...")
    try:
        while True:
            frame = receiver.get_latest_frame()
            if frame is None:
                continue
                
            start_t = time.perf_counter()
            
            # YOLO Inference (Resize implicitly handled by YOLO)
            results = model.predict(source=frame, imgsz=640, conf=0.5, verbose=False)
            
            corners, ball_center = get_platform_corners_and_ball(results)
            
            cam_x, cam_y = 0.0, 0.0
            
            if corners is not None and ball_center is not None:
                # Compute Homography
                src_pts = np.array(corners, dtype=np.float32)
                M, status = cv2.findHomography(src_pts, dst_pts)
                
                if M is not None:
                    # Transform ball center to mm
                    ball_pt = np.array([[[ball_center[0], ball_center[1]]]], dtype=np.float32)
                    ball_mm = cv2.perspectiveTransform(ball_pt, M)
                    cam_x, cam_y = ball_mm[0][0][0], ball_mm[0][0][1]
                    
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
            fps = 1.0 / (max(end_t - start_t, 0.001))
            
            # Visualization Phase
            disp = frame.copy()
            if corners is not None:
                for i in range(4):
                    p1 = tuple(map(int, corners[i]))
                    p2 = tuple(map(int, corners[(i+1)%4]))
                    cv2.line(disp, p1, p2, (255, 0, 0), 2)
                    cv2.circle(disp, p1, 5, (0, 0, 255), -1)
            
            if ball_center is not None:
                bc = tuple(map(int, ball_center))
                cv2.circle(disp, bc, 10, (0, 255, 255), -1)
                
            cv2.putText(disp, f"Cam: X={cam_x:.1f} Y={cam_y:.1f} mm", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(disp, f"FPS: {fps:.1f}", (20, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 200, 0), 2)
                        
            # Resize for viewing
            view_disp = cv2.resize(disp, (960, 540))
            cv2.imshow("Cascade Inference (Press 'q' to quit)", view_disp)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        receiver.stop()
        if ser:
            ser.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
