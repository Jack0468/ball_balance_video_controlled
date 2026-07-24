import cv2
import socket
import threading
import queue
import time
import struct
import numpy as np
import os
import sys
import serial
from ultralytics import YOLO

# Add parent directory to path to import ml_audio
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from ml_audio.audio_listener import AudioListener

# --- Configuration ---
UDP_IP = "0.0.0.0"
UDP_PORT = 5005
SERIAL_PORT = "COM3"
SERIAL_BAUD = 115200

TARGET_X = 0.0
TARGET_Y = 0.0

# Physical marker coordinates converted to center-origin (mm).
TARGET_COORDS = {
    "green": (-93.75 + 33.0, 71.0 - 26.0),
    "red": (93.75 - 41.0, 71.0 - 53.0),
    "yellow": (-93.75 + 69.0, -71.0 + 58.0),
    "blue": (93.75 - 13.0, -71.0 + 8.0),
}

# Physical Marker Coordinates in Platform-Centric Millimeters (Top-Left origin)
MARKERS_PHYSICAL_MM = np.array([
    [33.0, 26.0],                   # Green (Top-Left)
    [187.5 - 41.0, 53.0],           # Red (Top-Right)
    [69.0, 142.0 - 58.0],           # Grey (Bottom-Left)
    [187.5 - 13.0, 142.0 - 8.0]     # Black (Bottom-Right)
], dtype=np.float32)

class UDPReceiver:
    def __init__(self, ip, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((ip, port))
        self.frame_queue = queue.Queue(maxsize=1)
        self.running = True
        self.thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.thread.start()
        
    def _receive_loop(self):
        print(f"UDP Receiver listening on {UDP_IP}:{UDP_PORT}")
        while self.running:
            try:
                data, _ = self.sock.recvfrom(65535)
                frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
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

def order_markers(pts):
    # Sorts 4 points into: Top-Left, Top-Right, Bottom-Left, Bottom-Right
    pts = sorted(pts, key=lambda x: x[1]) # Sort by Y
    top = pts[:2]
    bottom = pts[2:]
    
    tl = min(top, key=lambda x: x[0])
    tr = max(top, key=lambda x: x[0])
    bl = min(bottom, key=lambda x: x[0])
    br = max(bottom, key=lambda x: x[0])
    
    return np.array([tl, tr, bl, br], dtype=np.float32)

def main():
    # 1. Model & Hardware Init
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'models/yolov8_marker_and_ball_detector/weights/best.pt')
    
    if not os.path.exists(model_path):
        print(f"ERROR: YOLO model not found at {model_path}. Train the model first.")
        return
        
    print("Loading YOLOv8 Model...")
    model = YOLO(model_path)
    
    try:
        ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0.1)
        print(f"Connected to STM32 on {SERIAL_PORT}")
    except serial.SerialException:
        print(f"WARNING: Could not open {SERIAL_PORT}. Running in vision-only mode.")
        ser = None

    receiver = UDPReceiver(UDP_IP, UDP_PORT)

    # Start background audio listener for command-driven target updates.
    audio = AudioListener()
    audio.start()
    
    # 2. Main Inference Loop
    print("Starting YOLO + Audio Inference Loop...")
    target_x = TARGET_X
    target_y = TARGET_Y
    try:
        while True:
            frame = receiver.get_latest_frame()
            if frame is None:
                continue
                
            start_t = time.perf_counter()
            
            # YOLO inference (imgsz=640 assumes standard stream)
            results = model.predict(source=frame, imgsz=640, conf=0.5, verbose=False)
            result = results[0]
            
            ball_pt = None
            marker_pts = []
            
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                
                if cls_id == 0: # Ball
                    ball_pt = (cx, cy)
                elif cls_id == 1: # Marker
                    marker_pts.append((cx, cy))
                    
            if len(marker_pts) >= 4 and ball_pt is not None:
                # We need exactly 4 for homography; take top 4 highest confidence if >4
                # But here we just take the first 4 for simplicity
                ordered_pixels = order_markers(marker_pts[:4])
                
                # M maps Pixels -> MM
                M, _ = cv2.findHomography(ordered_pixels, MARKERS_PHYSICAL_MM)
                
                if M is not None:
                    # Project ball to MM
                    b_px = np.array([[[ball_pt[0], ball_pt[1]]]], dtype=np.float32)
                    b_mm = cv2.perspectiveTransform(b_px, M)[0][0]
                    
                    # Convert to Platform Center Origin (assuming STM32 expects 0,0 at center)
                    # Platform W=187.5, H=142.0
                    final_x = b_mm[0] - (187.5 / 2.0)
                    final_y = b_mm[1] - (142.0 / 2.0)

                    # Update target from latest audio command state.
                    audio_state = audio.get_latest_command()
                    target_colour = audio_state.get("target_colour")
                    mode = audio_state.get("mode")

                    if mode == "colour_select" and target_colour in TARGET_COORDS:
                        target_x, target_y = TARGET_COORDS[target_colour]
                    elif mode == "hold":
                        pass
                    elif mode == "stop":
                        pass
                    
                    err_x = target_x - final_x
                    err_y = target_y - final_y
                    
                    # Send to STM32
                    if ser:
                        payload = struct.pack('<chh', b'<', int(err_x), int(err_y))
                        ser.write(payload)
                        
                    inference_ms = (time.perf_counter() - start_t) * 1000.0
                    print(
                        f"Ball: ({final_x:.1f}mm, {final_y:.1f}mm) | "
                        f"Target: ({target_x:.1f}, {target_y:.1f}) [{audio_state.get('command', 'hold')}] | "
                        f"Latency: {inference_ms:.1f}ms"
                    )
                    
            # (Optional) Display frame
            # annotated_frame = result.plot()
            # cv2.imshow("YOLO Tracker", annotated_frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
                
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        audio.stop()
        receiver.stop()
        if ser:
            ser.close()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
