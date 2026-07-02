import serial
import time
import struct
import numpy as np
import cv2
import os
import csv
import ok

class TrainingDataCollector:
    def __init__(self, teensy_port=None, baudrate=115200):
        # Opal Kelly FPGA Setup
        self.dev = ok.okCFrontPanel()
        if self.dev.NoError != self.dev.OpenBySerial(""):
            raise Exception("A device could not be opened. Is the XEM3010 connected?")
        
        # Configure FPGA with bitstream (assumes it's compiled, user must flash it manually or add code here)
        # self.dev.ConfigureFPGA("camera_streamer.bit")
        
        # We need to initialize the I2C configuration for the camera here
        self.init_camera_i2c()

        # Teensy Serial Setup (for sending motor commands)
        self.teensy_port = teensy_port
        if self.teensy_port:
            self.ser = serial.Serial(self.teensy_port, baudrate, timeout=1)
        else:
            self.ser = None
            print("Warning: No Teensy COM port provided. Motor commands will not be sent.")

        self.frame_width = 160
        self.frame_height = 120
        self.frame_size = self.frame_width * self.frame_height * 2 # RGB565 (2 bytes per pixel)
        self.sync_header = b'\xAA\xBB\xCC\xDD'
        
        # Payload = 4 bytes (sync) + 4 bytes (x, y) + 38400 bytes (frame)
        self.payload_size = 4 + 4 + self.frame_size
        
        # Setup Medallion Architecture Directories
        self.bronze_dir = os.path.join(os.path.dirname(__file__), "..", "ml_vision", "data", "bronze")
        self.frames_dir = os.path.join(self.bronze_dir, "frames")
        self.labels_file = os.path.join(self.bronze_dir, "labels.csv")
        
        os.makedirs(self.frames_dir, exist_ok=True)
        
        # Initialize CSV
        self.csv_exists = os.path.isfile(self.labels_file)
        self.frame_counter = 0

    def init_camera_i2c(self):
        """Bit-bangs the I2C configuration to the OV7670 via okWireIn."""
        print("Initializing Camera over I2C (Python Bit-bang)...")
        # I2C lines are on WireIn 0x00
        # Bit 0 = SCL
        # Bit 1 = SDA Out
        # Bit 2 = SDA OE (1 = Output, 0 = High-Z)
        # Bit 3 = Camera Reset (Active Low)
        # Bit 4 = ADC/FIFO Reset
        
        # Reset camera
        self.dev.SetWireInValue(0x00, 0x00, 0xFF) # Reset low
        self.dev.UpdateWireIns()
        time.sleep(0.1)
        self.dev.SetWireInValue(0x00, 0x09, 0xFF) # Reset high, SCL high
        self.dev.UpdateWireIns()
        time.sleep(0.1)
        
        # TODO: Port the OV7670_REG_SETUP array from C++ to a Python dictionary here
        # and implement a software I2C write function to configure the camera registers.
        # This allows rapid tweaking of brightness/contrast without recompiling Verilog.
        print("Camera reset complete. (Register configuration omitted for brevity)")

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
        if self.ser:
            command = f"T{target_x:.2f},{target_y:.2f}\n"
            self.ser.write(command.encode('utf-8'))

    def run(self):
        print("Starting Robust Active Data Collection via Opal Kelly. Press 'q' to quit.")
        import random
        
        def get_random_target():
            return random.uniform(-70, 70), random.uniform(-55, 55)

        frames_per_target = 100 
        
        # Buffer to read the full payload
        buf = bytearray(self.payload_size)
        
        with open(self.labels_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not self.csv_exists:
                writer.writerow(["filename", "timestamp_ms", "touch_x_mm", "touch_y_mm", "target_x", "target_y"])

            try:
                target_x, target_y = get_random_target()
                self.send_target(target_x, target_y)
                print(f"Initial target: ({target_x:.2f}, {target_y:.2f})")
                
                while True:
                    # Target Switching Logic
                    if self.frame_counter > 0 and self.frame_counter % frames_per_target == 0:
                        target_x, target_y = get_random_target()
                        self.send_target(target_x, target_y)
                        print(f"Switched target to: ({target_x:.2f}, {target_y:.2f})")

                    # Read exactly one payload size from Opal Kelly PipeOut endpoint 0xA0
                    bytes_read = self.dev.ReadFromPipeOut(0xA0, buf)
                    
                    if bytes_read == self.payload_size:
                        # 1. Verify Sync Header
                        if buf[0:4] != self.sync_header:
                            print("Warning: Sync header mismatch. Waiting for next frame.")
                            time.sleep(0.01) # Small delay to wait for VSYNC
                            continue
                            
                        # 2. Extract ADC X and Y (16-bit integers from the ADS1675)
                        # The FPGA sends Touch X as Word 2, Touch Y as Word 3
                        touch_x_raw, touch_y_raw = struct.unpack('>HH', buf[4:8])
                        
                        # Note: You may need to map the raw ADC integers to millimeters based on your calibration
                        touch_x = (touch_x_raw / 65535.0) * 140.0 - 70.0
                        touch_y = (touch_y_raw / 65535.0) * 110.0 - 55.0
                        
                        # 3. Extract Image Data
                        frame_data = buf[8:]
                        
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
                        print(f"Warning: Dropped payload. Expected {self.payload_size} bytes, got {bytes_read}.")
                        # Clear FIFO or wait a moment
                        time.sleep(0.05)
                        
            except KeyboardInterrupt:
                pass
            finally:
                print(f"Collection stopped. Saved {self.frame_counter} frames.")
                if self.ser:
                    self.ser.close()
                cv2.destroyAllWindows()

if __name__ == "__main__":
    # Teensy port for targets, Opal Kelly is auto-detected
    collector = TrainingDataCollector(teensy_port='COM3')
    collector.run()
