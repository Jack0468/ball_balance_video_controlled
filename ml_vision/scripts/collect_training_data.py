import serial
import time
import struct
import numpy as np
import cv2
import os
import csv

class TrainingDataCollector:
    def __init__(self, port, baudrate=2000000):
        self.ser = serial.Serial(port, baudrate, timeout=1)
        self.frame_width = 160
        self.frame_height = 120
        self.frame_size = self.frame_width * self.frame_height * 2 # RGB565 (2 bytes per pixel)
        self.sync_header = b'\xAA\xBB\xCC\xDD'
        
        # Payload = 4 bytes (sync) + 4 bytes (x) + 4 bytes (y) + 38400 bytes (frame)
        self.payload_size = 4 + 4 + self.frame_size
        
        # Setup Medallion Architecture Directories
        self.bronze_dir = os.path.join(os.path.dirname(__file__), "..", "data", "bronze")
        self.frames_dir = os.path.join(self.bronze_dir, "frames")
        self.labels_file = os.path.join(self.bronze_dir, "labels.csv")
        
        os.makedirs(self.frames_dir, exist_ok=True)
        
        # Initialize CSV
        self.csv_exists = os.path.isfile(self.labels_file)
        self.frame_counter = 0

    def rgb565_to_bgr(self, raw_bytes):
        """Converts raw RGB565 bytes to an OpenCV BGR image."""
        # Convert bytes to uint16 numpy array
        img16 = np.frombuffer(raw_bytes, dtype=np.uint16)
        img16 = img16.reshape((self.frame_height, self.frame_width))
        
        # Extract RGB channels
        r = ((img16 >> 11) & 0x1F) * 255 // 31
        g = ((img16 >> 5) & 0x3F) * 255 // 63
        b = (img16 & 0x1F) * 255 // 31
        
        # Stack into BGR for OpenCV
        bgr_img = np.dstack((b, g, r)).astype(np.uint8)
        return bgr_img

    def send_target(self, target_x, target_y):
        """Sends the dummy target to the Teensy to control the PID loop."""
        command = f"T{target_x:.2f},{target_y:.2f}\n"
        self.ser.write(command.encode('utf-8'))

    def run(self):
        print("Starting Robust Active Data Collection. Press 'q' to quit.")
        
        # We will use random uniform sampling across the platform's bounding box
        # to maximize the state-space exploration for the vision model.
        # Platform is approx 167x135mm -> Safe bounds: X[-70, 70], Y[-55, 55]
        import random
        
        def get_random_target():
            return random.uniform(-70, 70), random.uniform(-55, 55)

        frames_per_target = 100 # Change target every 100 frames to allow settling time
        
        with open(self.labels_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not self.csv_exists:
                writer.writerow(["filename", "timestamp_ms", "touch_x_mm", "touch_y_mm", "target_x", "target_y"])

            try:
                # Send the initial random target
                target_x, target_y = get_random_target()
                self.send_target(target_x, target_y)
                print(f"Initial target: ({target_x:.2f}, {target_y:.2f})")
                
                while True:
                    try:
                        # Target Switching Logic
                        if self.frame_counter > 0 and self.frame_counter % frames_per_target == 0:
                            target_x, target_y = get_random_target()
                            self.send_target(target_x, target_y)
                            print(f"Switched target to: ({target_x:.2f}, {target_y:.2f})")

                        # 1. Align to sync header robustly
                        sync_buffer = bytearray()
                        sync_timeout = time.time() + 2.0 # 2 seconds timeout for sync
                        
                        while True:
                            if time.time() > sync_timeout:
                                print("Warning: Sync header timeout, flushing serial buffer.")
                                self.ser.reset_input_buffer()
                                break
                                
                            if self.ser.in_waiting > 0:
                                sync_buffer.extend(self.ser.read(1))
                                if len(sync_buffer) >= 4:
                                    if sync_buffer[-4:] == self.sync_header:
                                        break
                                    # Keep buffer small to avoid memory leak during garbage data
                                    if len(sync_buffer) > 1024:
                                        sync_buffer = sync_buffer[-4:]
                        
                        if time.time() > sync_timeout:
                            continue # Try syncing again on the next iteration
                        
                        # 2. Read the remaining payload (x, y, frame)
                        payload_remainder_size = 8 + self.frame_size
                        
                        # Use a blocking read to ensure we get exactly the right amount of bytes
                        # The serial timeout handles disconnecting if the bot crashes
                        payload = self.ser.read(payload_remainder_size)
                        
                        if len(payload) == payload_remainder_size:
                            # Unpack float32 x and y (little-endian)
                            touch_x, touch_y = struct.unpack('<ff', payload[:8])
                            frame_data = payload[8:]
                            
                            # Process Image
                            bgr_img = self.rgb565_to_bgr(frame_data)
                            
                            # Save Image & Label
                            timestamp_ms = int(time.time() * 1000)
                            filename = f"frame_{timestamp_ms}_{self.frame_counter:05d}.png"
                            filepath = os.path.join(self.frames_dir, filename)
                            
                            cv2.imwrite(filepath, bgr_img)
                            writer.writerow([filename, timestamp_ms, touch_x, touch_y, target_x, target_y])
                            
                            self.frame_counter += 1
                            
                            # Visualization
                            display_img = cv2.resize(bgr_img, (640, 480)) # upscale for visibility
                            cv2.putText(display_img, f"Pos: {touch_x:.1f}, {touch_y:.1f}", (10, 30), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            cv2.putText(display_img, f"Tgt: {target_x:.1f}, {target_y:.1f}", (10, 70), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            
                            cv2.imshow("Data Collection Feed", display_img)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break
                        else:
                            print(f"Warning: Dropped payload. Expected {payload_remainder_size} bytes, got {len(payload)}.")
                            self.ser.reset_input_buffer()
                            
                    except serial.SerialException as e:
                        print(f"Serial Exception: {e}. Attempting to reconnect...")
                        time.sleep(2)
                        try:
                            self.ser.close()
                            self.ser.open()
                        except:
                            pass
                            
            except KeyboardInterrupt:
                pass
            finally:
                print(f"Collection stopped. Saved {self.frame_counter} frames.")
                self.ser.close()
                cv2.destroyAllWindows()

if __name__ == "__main__":
    collector = TrainingDataCollector(port='COM3')
    collector.run()
