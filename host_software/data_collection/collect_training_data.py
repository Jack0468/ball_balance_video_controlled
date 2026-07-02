import time
import struct
import numpy as np
import cv2
import os
import csv
import math
import random

# Try to import Opal Kelly. Mock it if not available for testing.
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
                # Mock a frame payload
                buf[0:4] = b'\xAA\xBB\xCC\xDD'
                return len(buf)

# Must import correctly depending on execution context, try local if package not found
try:
    from host_software.control.inverse_kinematics import get_target_angles
    from host_software.control.pid_controller import PIDController
except ImportError:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from host_software.control.inverse_kinematics import get_target_angles
    from host_software.control.pid_controller import PIDController

STEPS_PER_DEGREE = 3200 / 360.0
PLATFORM_HEIGHT = 87.0 # mm

class DataCollectionStateMachine:
    def __init__(self):
        self.state = "PHASE_1_RANDOM"
        self.frame_count = 0
        self.phase1_duration = 3000 # 3000 frames (~100 seconds)
        
        self.phase2_patterns = ["figure8", "spiral"]
        self.current_pattern_idx = 0
        self.phase2_frames_per_pattern = 1500 # ~50 seconds per pattern
        
        # Grid of points for phase 3 sweeps
        self.start_points = []
        for x in range(-40, 41, 40): # -40, 0, 40
            for y in range(-30, 31, 30): # -30, 0, 30
                self.start_points.append((x, y))
        self.current_start_idx = 0
        
        self.directions = [
            (0, 1), (0, -1), (-1, 0), (1, 0),
            (1, 1), (-1, -1), (-1, 1), (1, -1)
        ]
        self.current_dir_idx = 0
        self.sweep_distance = 0
        
        self.target_x = 0.0
        self.target_y = 0.0
        
    def get_next_target(self):
        self.frame_count += 1
        is_done = False
        
        if self.state == "PHASE_1_RANDOM":
            if self.frame_count % 100 == 1:
                self.target_x = random.uniform(-60, 60)
                self.target_y = random.uniform(-45, 45)
                
            if self.frame_count > self.phase1_duration:
                self.state = "PHASE_2_PATTERNS"
                self.frame_count = 0
                print("Transitioning to PHASE_2_PATTERNS")
                
        elif self.state == "PHASE_2_PATTERNS":
            pattern = self.phase2_patterns[self.current_pattern_idx]
            t = self.frame_count * 0.05
            
            if pattern == "figure8":
                self.target_x = 50 * math.sin(t)
                self.target_y = 30 * math.sin(2*t)
            elif pattern == "spiral":
                radius = 5 + 40 * (self.frame_count / self.phase2_frames_per_pattern)
                self.target_x = radius * math.cos(t * 1.5)
                self.target_y = radius * math.sin(t * 1.5)
                
            if self.frame_count > self.phase2_frames_per_pattern:
                self.current_pattern_idx += 1
                self.frame_count = 0
                if self.current_pattern_idx >= len(self.phase2_patterns):
                    self.state = "PHASE_3_SWEEPS"
                    print("Transitioning to PHASE_3_SWEEPS")
                    
        elif self.state == "PHASE_3_SWEEPS":
            if self.current_start_idx >= len(self.start_points):
                is_done = True
                return self.target_x, self.target_y, is_done
                
            start_x, start_y = self.start_points[self.current_start_idx]
            dir_dx, dir_dy = self.directions[self.current_dir_idx]
            
            # Normalize direction vector
            mag = math.sqrt(dir_dx**2 + dir_dy**2)
            dir_dx /= mag
            dir_dy /= mag
            
            # Every 10 frames, move 5mm
            if self.frame_count % 10 == 0:
                self.sweep_distance += 5
                
            self.target_x = start_x + dir_dx * self.sweep_distance
            self.target_y = start_y + dir_dy * self.sweep_distance
            
            # If we hit the bounds of the plate, go to next direction
            if abs(self.target_x) > 65 or abs(self.target_y) > 50 or self.sweep_distance >= 100:
                self.sweep_distance = 0
                self.current_dir_idx += 1
                if self.current_dir_idx >= len(self.directions):
                    self.current_dir_idx = 0
                    self.current_start_idx += 1
                    
        return self.target_x, self.target_y, is_done

class TrainingDataCollector:
    def __init__(self):
        # Opal Kelly FPGA Setup
        self.dev = ok.okCFrontPanel()
        if self.dev.NoError != self.dev.OpenBySerial(""):
            print("WARNING: A device could not be opened. Is the XEM3010 connected?")
            
        print("Configuring PLL22393 for robust hardware clock (clk1)...")
        pll = ok.PLL22393()
        pll.SetReference(48.0)                 # 48 MHz crystal reference on the XEM3010
        pll.SetPLLParameters(0, 48, 48, True)  # PLL0 = 48 MHz
        pll.SetOutputSource(0, ok.PLL22393.ClkSrc_PLL0_0)
        pll.SetOutputDivider(0, 1)             # 48 MHz / 1 = 48 MHz on clk1
        pll.SetOutputEnable(0, True)
        self.dev.SetPLL22393Configuration(pll) 
        
        self.pid = PIDController()
        self.state_machine = DataCollectionStateMachine()
        
        self.init_camera_i2c()

        self.frame_width = 160
        self.frame_height = 120
        self.frame_size = self.frame_width * self.frame_height * 2 # RGB565 (2 bytes per pixel)
        self.sync_header = b'\xAA\xBB\xCC\xDD'
        
        # Payload = 4 bytes (sync) + 4 bytes (x, y) + 38400 bytes (frame)
        self.payload_size = 4 + 4 + self.frame_size
        
        # Setup Medallion Architecture Directories
        self.bronze_dir = os.path.join(os.path.dirname(__file__), "..", "..", "ml_vision", "data", "bronze")
        self.frames_dir = os.path.join(self.bronze_dir, "frames")
        self.labels_file = os.path.join(self.bronze_dir, "labels.csv")
        
        os.makedirs(self.frames_dir, exist_ok=True)
        
        # Initialize CSV
        self.csv_exists = os.path.isfile(self.labels_file)
        self.frames_saved = 0

    def init_camera_i2c(self):
        """Bit-bangs the I2C configuration to the OV7670 via okWireIn."""
        print("Initializing Camera over I2C (Python Bit-bang)...")
        # Reset camera
        self.dev.SetWireInValue(0x00, 0x00, 0xFF)
        self.dev.UpdateWireIns()
        time.sleep(0.1)
        self.dev.SetWireInValue(0x00, 0x09, 0xFF)
        self.dev.UpdateWireIns()
        time.sleep(0.1)
        
    def angle_to_steps(self, angle_deg):
        return int(round(angle_deg * STEPS_PER_DEGREE))

    def update_motors(self, theta_a, theta_b, theta_c):
        """Sends the exact target steps to the FPGA and triggers an atomic update."""
        steps_a = self.angle_to_steps(theta_a)
        steps_b = self.angle_to_steps(theta_b)
        steps_c = self.angle_to_steps(theta_c)
        
        def split_32(val):
            val = val & 0xFFFFFFFF
            return val & 0xFFFF, (val >> 16) & 0xFFFF
            
        a_lsb, a_msb = split_32(steps_a)
        b_lsb, b_msb = split_32(steps_b)
        c_lsb, c_msb = split_32(steps_c)
        
        self.dev.SetWireInValue(0x01, a_lsb, 0xFFFF)
        self.dev.SetWireInValue(0x02, a_msb, 0xFFFF)
        self.dev.SetWireInValue(0x03, b_lsb, 0xFFFF)
        self.dev.SetWireInValue(0x04, b_msb, 0xFFFF)
        self.dev.SetWireInValue(0x05, c_lsb, 0xFFFF)
        self.dev.SetWireInValue(0x06, c_msb, 0xFFFF)
        self.dev.UpdateWireIns()
        
        # Trigger the atomic latch
        self.dev.ActivateTriggerIn(0x40, 0)

    def rgb565_to_bgr(self, raw_bytes):
        """Converts raw RGB565 bytes to an OpenCV BGR image."""
        img16 = np.frombuffer(raw_bytes, dtype=np.uint16)
        img16 = img16.reshape((self.frame_height, self.frame_width))
        r = ((img16 >> 11) & 0x1F) * 255 // 31
        g = ((img16 >> 5) & 0x3F) * 255 // 63
        b = (img16 & 0x1F) * 255 // 31
        bgr_img = np.dstack((b, g, r)).astype(np.uint8)
        return bgr_img

    def run(self):
        print("Starting Robust Active Data Collection via Opal Kelly. Press 'q' to quit.")
        
        buf = bytearray(self.payload_size)
        
        with open(self.labels_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not self.csv_exists:
                writer.writerow(["filename", "timestamp_ms", "touch_x_mm", "touch_y_mm", "target_x", "target_y"])

            try:
                while True:
                    # Read exactly one payload size from Opal Kelly PipeOut endpoint 0xA0
                    # This blocking read automatically syncs our control loop to the camera frame rate (30fps)
                    bytes_read = self.dev.ReadFromPipeOut(0xA0, buf)
                    
                    if bytes_read == self.payload_size:
                        if buf[0:4] != self.sync_header:
                            print("Warning: Sync header mismatch. Waiting for next frame.")
                            time.sleep(0.01)
                            continue
                            
                        # Extract Touch Coordinates (16-bit raw ADC)
                        touch_x_raw, touch_y_raw = struct.unpack('>HH', buf[4:8])
                        touch_x = (touch_x_raw / 65535.0) * 140.0 - 70.0
                        touch_y = (touch_y_raw / 65535.0) * 110.0 - 55.0
                        
                        # --- MOTOR CONTROL LOOP ---
                        target_x, target_y, is_done = self.state_machine.get_next_target()
                        if is_done:
                            print("Data Collection Completed All Phases!")
                            break
                            
                        # 1. Run PID
                        pitch, roll = self.pid.calculate_angles(target_x, target_y, touch_x, touch_y)
                        # 2. Run IK
                        theta_a, theta_b, theta_c = get_target_angles(pitch, -roll, PLATFORM_HEIGHT)
                        # 3. Update Hardware
                        self.update_motors(theta_a, theta_b, theta_c)
                        
                        # --- DATA LOGGING ---
                        frame_data = buf[8:]
                        bgr_img = self.rgb565_to_bgr(frame_data)
                        
                        timestamp_ms = int(time.time() * 1000)
                        filename = f"frame_{timestamp_ms}_{self.frames_saved:05d}.png"
                        filepath = os.path.join(self.frames_dir, filename)
                        
                        cv2.imwrite(filepath, bgr_img)
                        writer.writerow([filename, timestamp_ms, touch_x, touch_y, target_x, target_y])
                        self.frames_saved += 1
                        
                        # Visualization
                        display_img = cv2.resize(bgr_img, (640, 480))
                        cv2.putText(display_img, f"State: {self.state_machine.state}", (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        cv2.putText(display_img, f"Pos: {touch_x:.1f}, {touch_y:.1f}", (10, 70), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(display_img, f"Tgt: {target_x:.1f}, {target_y:.1f}", (10, 110), 
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
                print(f"Collection stopped. Saved {self.frames_saved} frames.")
                cv2.destroyAllWindows()

if __name__ == "__main__":
    collector = TrainingDataCollector()
    collector.run()
