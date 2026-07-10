import time
import struct
import numpy as np
import cv2
import os
import csv
import argparse
import threading
import serial

class IphoneDataLogger:
    def __init__(self, port, baudrate=2000000):
        self.port = port
        self.baudrate = baudrate
        
        # Setup Medallion Architecture Directories
        self.bronze_dir = os.path.join(os.path.dirname(__file__), "..", "..", "ml_vision", "data", "bronze")
        self.telemetry_file = os.path.join(self.bronze_dir, "iphone_telemetry.csv")
        
        os.makedirs(self.bronze_dir, exist_ok=True)
        self.telemetry_csv_exists = os.path.isfile(self.telemetry_file)
        
        self.latest_telemetry = None
        self.telemetry_lock = threading.Lock()
        
        self.stop_event = threading.Event()
        self.telemetry_thread = threading.Thread(target=self.read_serial_telemetry)
        self.telemetry_thread.daemon = True

        self.last_sync_color = (255, 255, 255)

    def read_serial_telemetry(self):
        print(f"Connecting to STM32 on {self.port} at {self.baudrate} baud...")
        try:
            ser = serial.Serial(self.port, self.baudrate)
        except Exception as e:
            print(f"Error opening serial port: {e}")
            return
            
        # Format: teensy_micros(I) + 15 floats(f) = 64 bytes total (excluding sync header)
        struct_format = "<Ifffffffffffffff"
        expected_size = struct.calcsize(struct_format)
        
        with open(self.telemetry_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not self.telemetry_csv_exists:
                writer.writerow(["host_timestamp_ms", "mcu_micros", "target_x", "target_y", 
                                 "touch_x", "touch_y", "error_x", "error_y", "pitch", "roll", 
                                 "theta_a", "theta_b", "theta_c", "integral_x", "integral_y", 
                                 "deriv_x", "deriv_y"])
            
            sync_buf = bytearray()
            while not self.stop_event.is_set():
                if ser.in_waiting > 0:
                    b = ser.read(1)
                    sync_buf.append(b[0])
                    if len(sync_buf) > 4:
                        sync_buf.pop(0)
                        
                    # Match sync header 0xAABBCCDD
                    if bytes(sync_buf) == b'\xAA\xBB\xCC\xDD':
                        data = ser.read(expected_size)
                        if len(data) == expected_size:
                            host_time_ms = int(time.time() * 1000)
                            unpacked = struct.unpack(struct_format, data)
                            
                            with self.telemetry_lock:
                                self.latest_telemetry = (host_time_ms,) + unpacked
                            
                            # Write to CSV
                            writer.writerow(self.latest_telemetry)
                            f.flush()
                        sync_buf.clear()

    def run(self):
        self.telemetry_thread.start()
        
        print("Starting Visual Sync Display. Point your iPhone at the screen AND the robot.")
        print("Press 'q' to quit.")
        print(f"NOTE: Data is logging to {self.telemetry_file}")
        print("NOTE: Old plot_log.py will not work with this binary stream!")
        
        try:
            while True:
                host_timestamp_ms = int(time.time() * 1000)
                
                # Grab the most recent telemetry
                with self.telemetry_lock:
                    current_tel = self.latest_telemetry
                
                # Create a massive blank image
                display_img = np.zeros((720, 1280, 3), dtype=np.uint8)
                
                if current_tel:
                    # Every 500ms, flash the background color slightly to create a strong visual sync pulse
                    if (host_timestamp_ms // 500) % 2 == 0:
                        display_img[:] = (30, 30, 30) # Dark gray
                    else:
                        display_img[:] = (0, 0, 0) # Black
                        
                    target_x = current_tel[2]
                    target_y = current_tel[3]
                    touch_x = current_tel[4]
                    touch_y = current_tel[5]
                    
                    # Draw giant timestamp
                    cv2.putText(display_img, f"{host_timestamp_ms}", (50, 300), 
                                cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 10)
                                
                    cv2.putText(display_img, "TIMESTAMP (MS)", (50, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 200, 0), 4)
                                
                    # Draw telemetry info
                    cv2.putText(display_img, f"Pos: {touch_x:.1f}, {touch_y:.1f}", (50, 500), 
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
                    cv2.putText(display_img, f"Tgt: {target_x:.1f}, {target_y:.1f}", (50, 600), 
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 150, 255), 3)
                else:
                    cv2.putText(display_img, "WAITING FOR STM32...", (50, 300), 
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
                
                cv2.imshow("iPhone Sync Display (POINT CAMERA HERE)", display_img)
                
                if cv2.waitKey(16) & 0xFF == ord('q'): # ~60fps refresh for the screen
                    break
                    
        except KeyboardInterrupt:
            pass
        finally:
            self.stop_event.set()
            print("Collection stopped.")
            cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Collect telemetry and show sync screen.')
    parser.add_argument('--port', type=str, required=True, help='STM32 COM port (e.g. COM8 or /dev/ttyACM0)')
    parser.add_argument('--baud', type=int, default=2000000, help='Baud rate')
    args = parser.parse_args()
    
    logger = IphoneDataLogger(args.port, args.baud)
    logger.run()
