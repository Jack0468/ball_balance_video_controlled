import cv2
import threading
import queue
import time
import struct
import numpy as np
import os
import argparse
import serial
import csv
from ultralytics import YOLO

# --- Configuration ---
SERIAL_PORT = "COM8"
SERIAL_BAUD = 2000000
LOG_FILE = "laptop_camera_telemetry_yolo.csv"

# Physical dimensions of the purple board (in mm)
PLATFORM_W = 187.5
PLATFORM_H = 142.0

# The physical corners of the board (Top-Left origin)
BOARD_CORNERS_MM = np.array([
    [0.0, 0.0],                     # Top-Left
    [PLATFORM_W, 0.0],              # Top-Right
    [PLATFORM_W, PLATFORM_H],       # Bottom-Right
    [0.0, PLATFORM_H]               # Bottom-Left
], dtype=np.float32)
# ---------------------

class TelemetryLogger:
    def __init__(self, serial_port):
        self.ser = serial_port
        self.running = True
        self.log_queue = queue.Queue()
        
        self.csv_file = open(LOG_FILE, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "host_time_ms", "mcu_micros", "target_x", "target_y", 
            "touch_x", "touch_y", "cam_x", "cam_y", "error_x", "error_y",
            "pitch", "roll", "theta_a", "theta_b", "theta_c",
            "integral_x", "integral_y", "deriv_x", "deriv_y"
        ])
        
        self.struct_fmt = '<IIfffffffffffffff'
        self.struct_len = struct.calcsize(self.struct_fmt)
        self.sync_word = b'\xAA\xBB\xCC\xDD'
        
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
                        sync_idx = buffer.find(self.sync_word)
                        if sync_idx == -1:
                            buffer.clear()
                            break
                        if sync_idx + self.struct_len <= len(buffer):
                            packet = buffer[sync_idx : sync_idx + self.struct_len]
                            self._parse_packet(packet)
                            buffer = buffer[sync_idx + self.struct_len :]
                        else:
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
            host_time = int(time.time() * 1000)
            self.csv_writer.writerow([
                host_time, unpacked[1], unpacked[2], unpacked[3], 
                unpacked[4], unpacked[5], self.latest_cam_x, self.latest_cam_y,
                unpacked[6], unpacked[7], unpacked[8], unpacked[9], 
                unpacked[10], unpacked[11], unpacked[12],
                unpacked[13], unpacked[14], unpacked[15], unpacked[16]
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

def order_corners(pts):
    pts = sorted(pts, key=lambda x: x[1])
    top = pts[:2]
    bottom = pts[2:]
    tl = min(top, key=lambda x: x[0])
    tr = max(top, key=lambda x: x[0])
    bl = min(bottom, key=lambda x: x[0])
    br = max(bottom, key=lambda x: x[0])
    return np.array([tl, tr, br, bl], dtype=np.float32)

def main():
    parser = argparse.ArgumentParser(description="Laptop Camera YOLO Inference")
    parser.add_argument("--cam_id", type=int, default=0, help="USB Camera ID")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load Unified YOLO-Pose Model
    model_path = os.path.join(script_dir, 'models/unified_pose_model/weights/best.pt')
    if not os.path.exists(model_path):
        print(f"ERROR: Unified YOLO-Pose model not found at {model_path}. Train the model first.")
        return
        
    print("Loading Unified YOLOv8-Pose Model...")
    model = YOLO(model_path)
    
    try:
        ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0.1)
        print(f"Connected to STM32 on {SERIAL_PORT}")
    except serial.SerialException:
        print(f"WARNING: Could not open {SERIAL_PORT}. Running in vision-only mode.")
        ser = None

    receiver = USBReceiver(args.cam_id)
    logger = TelemetryLogger(ser)
    
    print("Starting Unified YOLO Inference Loop...")
    try:
        while True:
            frame = receiver.get_latest_frame()
            if frame is None:
                continue
                
            start_t = time.perf_counter()
            
            # Inference pass (detects platform + ball + targets simultaneously)
            results = model.predict(source=frame, imgsz=640, conf=0.5, verbose=False)
            result = results[0]
            annotated_frame = result.plot()
            
            # Track identified objects
            platform_kpts = None
            ball_px = None
            target_pxs = []
            
            # Find the platform keypoints
            for idx, box in enumerate(result.boxes):
                cls_id = int(box.cls[0].item())
                if cls_id == 0: # Platform
                    if result.keypoints is not None and len(result.keypoints.xy) > idx:
                        kpts = result.keypoints.xy[idx].cpu().numpy()
                        if len(kpts) >= 4:
                            platform_kpts = order_corners(kpts[:4])
                elif cls_id == 1: # Ball
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    ball_px = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                else: # Targets (2,3,4,5)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    target_pxs.append((cls_id, (x1 + x2) / 2.0, (y1 + y2) / 2.0))
                    
            # Compute dynamic homography and project
            if platform_kpts is not None:
                # MM -> Pixels homography
                M, _ = cv2.findHomography(BOARD_CORNERS_MM, platform_kpts)
                
                if M is not None:
                    # Inverse Homography (Pixels -> MM)
                    M_inv = np.linalg.inv(M)
                    
                    if ball_px is not None:
                        # Project ball to MM
                        b_px_arr = np.array([[[ball_px[0], ball_px[1]]]], dtype=np.float32)
                        b_mm = cv2.perspectiveTransform(b_px_arr, M_inv)[0][0]
                        
                        # Convert to Platform Center Origin
                        final_x = b_mm[0] - (PLATFORM_W / 2.0)
                        final_y = b_mm[1] - (PLATFORM_H / 2.0)
                        
                        logger.update_cam_pos(final_x, final_y)
                        
                        # Send to STM32
                        if ser:
                            try:
                                cam_x_int = int(max(min(final_x, 32767), -32768))
                                cam_y_int = int(max(min(final_y, 32767), -32768))
                                payload = struct.pack('<chh', b'<', cam_x_int, cam_y_int)
                                ser.write(payload)
                            except Exception as e:
                                print(f"Serial Error: {e}")
                                
                        cv2.putText(annotated_frame, f"Cam: X={final_x:.1f} Y={final_y:.1f} mm", (20, 90), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Compute and display dynamic targets (optional, future use)
                    for cls_id, t_px in target_pxs:
                        t_px_arr = np.array([[[t_px[0], t_px[1]]]], dtype=np.float32)
                        t_mm = cv2.perspectiveTransform(t_px_arr, M_inv)[0][0]
                        tx = t_mm[0] - (PLATFORM_W / 2.0)
                        ty = t_mm[1] - (PLATFORM_H / 2.0)
                        cv2.putText(annotated_frame, f"T{cls_id}: {tx:.0f},{ty:.0f}mm", (int(t_px[0]), int(t_px[1]-10)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            else:
                cv2.putText(annotated_frame, "NO PLATFORM DETECTED", (20, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        
            end_t = time.perf_counter()
            fps = 1.0 / (end_t - start_t)
            
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 200, 0), 2)
            cv2.imshow("Unified YOLO-Pose Tracker", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        receiver.stop()
        logger.stop()
        if ser:
            ser.close()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
