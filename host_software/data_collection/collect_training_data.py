import time
import struct
import numpy as np
import cv2
import os
import csv
import argparse
import threading
import serial

try:
    import ok
except ImportError:
    print("WARNING: ok (Opal Kelly) library not found. Running in mock mode.")
    class ok:
        class okCFrontPanel:
            def OpenBySerial(self, serial): return 0
            def ConfigureFPGA(self, bitfile): return 0
            def SetWireInValue(self, addr, val, mask): pass
            def UpdateWireIns(self): pass
            def UpdateWireOuts(self): pass
            def GetWireOutValue(self, addr): return 0
            def ActivateTriggerIn(self, addr, bit): pass
            def ReadFromPipeOut(self, addr, buf):
                buf[0:4] = b'\xAA\xBB\xCC\xDD'
                return len(buf)

class TrainingDataCollector:
    def __init__(self, teensy_port):
        self.teensy_port = teensy_port
        
        # Opal Kelly FPGA Setup
        self.dev = ok.okCFrontPanel()
        if self.dev.NoError != self.dev.OpenBySerial(""):
            print("WARNING: A device could not be opened. Is the XEM3010 connected?")
            
        print("Configuring PLL22393 for robust hardware clock (clk1)...")
        pll = ok.PLL22393()
        pll.SetReference(48.0)
        pll.SetPLLParameters(0, 48, 48, True)
        pll.SetOutputSource(0, ok.PLL22393.ClkSrc_PLL0_0)
        pll.SetOutputDivider(0, 1)
        pll.SetOutputEnable(0, True)
        self.dev.SetPLL22393Configuration(pll) 
        
        self.init_camera_i2c()

        self.frame_width = 160
        self.frame_height = 120
        self.frame_size = self.frame_width * self.frame_height * 2 # RGB565
        self.sync_header = b'\xAA\xBB\xCC\xDD'
        
        # Keep original payload size to not break FPGA, just ignore the 4 touch bytes
        self.payload_size = 4 + 4 + self.frame_size
        
        # Setup Medallion Architecture Directories
        self.bronze_dir = os.path.join(os.path.dirname(__file__), "..", "..", "ml_vision", "data", "bronze")
        self.frames_dir = os.path.join(self.bronze_dir, "frames")
        self.images_file = os.path.join(self.bronze_dir, "images.csv")
        self.telemetry_file = os.path.join(self.bronze_dir, "telemetry.csv")
        
        os.makedirs(self.frames_dir, exist_ok=True)
        
        self.images_csv_exists = os.path.isfile(self.images_file)
        self.telemetry_csv_exists = os.path.isfile(self.telemetry_file)
        self.frames_saved = 0
        
        self.latest_telemetry = None
        self.telemetry_lock = threading.Lock()
        
        self.stop_event = threading.Event()
        self.telemetry_thread = threading.Thread(target=self.read_teensy_telemetry)
        self.telemetry_thread.daemon = True

    def init_camera_i2c(self):
        print("Initializing Camera over I2C (Python Bit-bang)...")
        self.dev.SetWireInValue(0x00, 0x00, 0xFF)
        self.dev.UpdateWireIns()
        time.sleep(0.1)
        self.dev.SetWireInValue(0x00, 0x09, 0xFF)
        self.dev.UpdateWireIns()
        time.sleep(0.1)

    def rgb565_to_bgr(self, raw_bytes):
        img16 = np.frombuffer(raw_bytes, dtype=np.uint16)
        img16 = img16.reshape((self.frame_height, self.frame_width))
        r = ((img16 >> 11) & 0x1F) * 255 // 31
        g = ((img16 >> 5) & 0x3F) * 255 // 63
        b = (img16 & 0x1F) * 255 // 31
        bgr_img = np.dstack((b, g, r)).astype(np.uint8)
        return bgr_img

    def read_teensy_telemetry(self):
        print(f"Connecting to Teensy on {self.teensy_port}...")
        try:
            ser = serial.Serial(self.teensy_port, 2000000)
        except Exception as e:
            print(f"Error opening Teensy serial port: {e}")
            return
            
        # Format: teensy_micros(I) + 15 floats(f) = 64 bytes total (excluding sync header)
        struct_format = "<Ifffffffffffffff"
        expected_size = struct.calcsize(struct_format)
        
        with open(self.telemetry_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not self.telemetry_csv_exists:
                writer.writerow(["host_timestamp_ms", "teensy_micros", "target_x", "target_y", 
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
                        sync_buf.clear()

    def run(self):
        self.telemetry_thread.start()
        
        print("Starting Robust Active Data Collection. Press 'q' to quit.")
        buf = bytearray(self.payload_size)
        
        with open(self.images_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not self.images_csv_exists:
                writer.writerow(["filename", "host_timestamp_ms", "teensy_micros", "target_x", "target_y", 
                                 "touch_x", "touch_y", "error_x", "error_y", "pitch", "roll", 
                                 "theta_a", "theta_b", "theta_c", "integral_x", "integral_y", 
                                 "deriv_x", "deriv_y"])

            try:
                while True:
                    # Blocking read from Opal Kelly PipeOut endpoint 0xA0
                    bytes_read = self.dev.ReadFromPipeOut(0xA0, buf)
                    
                    if bytes_read == self.payload_size:
                        if buf[0:4] != self.sync_header:
                            print("Warning: Sync header mismatch on FPGA read. Waiting for next frame.")
                            time.sleep(0.01)
                            continue
                            
                        host_timestamp_ms = int(time.time() * 1000)
                        
                        # Grab the most recent telemetry
                        with self.telemetry_lock:
                            current_tel = self.latest_telemetry
                        
                        # --- DATA LOGGING ---
                        frame_data = buf[8:]
                        bgr_img = self.rgb565_to_bgr(frame_data)
                        
                        filename = f"frame_{host_timestamp_ms}_{self.frames_saved:05d}.png"
                        filepath = os.path.join(self.frames_dir, filename)
                        cv2.imwrite(filepath, bgr_img)
                        
                        if current_tel:
                            # current_tel contains host_time_ms as first element, we just replace it with the image's timestamp or keep both
                            row = [filename, host_timestamp_ms] + list(current_tel[1:])
                            writer.writerow(row)
                            
                            target_x = current_tel[2]
                            target_y = current_tel[3]
                            touch_x = current_tel[4]
                            touch_y = current_tel[5]
                        else:
                            # Fallback if no telemetry yet
                            row = [filename, host_timestamp_ms] + [0]*16
                            writer.writerow(row)
                            target_x = target_y = touch_x = touch_y = 0.0
                            
                        self.frames_saved += 1
                        
                        # Visualization
                        display_img = cv2.resize(bgr_img, (640, 480))
                        cv2.putText(display_img, f"Pos: {touch_x:.1f}, {touch_y:.1f}", (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(display_img, f"Tgt: {target_x:.1f}, {target_y:.1f}", (10, 70), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        
                        cv2.imshow("Data Collection Feed", display_img)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    else:
                        print(f"Warning: Dropped payload. Expected {self.payload_size} bytes, got {bytes_read}.")
                        time.sleep(0.05)
                        
            except KeyboardInterrupt:
                pass
            finally:
                self.stop_event.set()
                print(f"Collection stopped. Saved {self.frames_saved} frames.")
                cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Collect training data.')
    parser.add_argument('--port', type=str, required=True, help='Teensy COM port (e.g. COM3 or /dev/ttyACM0)')
    args = parser.parse_args()
    
    collector = TrainingDataCollector(args.port)
    collector.run()
