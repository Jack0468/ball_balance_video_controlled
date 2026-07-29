import cv2
import threading
import queue
import time
import socket
import struct
import numpy as np

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
