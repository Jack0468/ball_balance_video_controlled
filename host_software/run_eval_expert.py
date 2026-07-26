import time
import csv
import os
import struct

# Dummy imports to illustrate the architecture - adapt to your specific serial/vision modules
# from src.openvino_dispatcher import OpenVINODispatcher
# from src.serial_interface import SerialConnection

class ExpertEvaluator:
    def __init__(self, port="COM3", baud=115200):
        print(f"Connecting to Arduino on {port}...")
        # self.serial = SerialConnection(port, baud)
        # self.vision = OpenVINODispatcher("models/yolov8_marker_best.xml")
        
        self.output_dir = "evaluations/04_evaluation"
        os.makedirs(self.output_dir, exist_ok=True)
        self.csv_file = open(os.path.join(self.output_dir, "labels_sequential_expert.csv"), "w", newline="")
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow(["host_timestamp_ms", "target_x", "target_y", "touch_x", "touch_y", "theta_a", "theta_b", "theta_c"])
        
    def get_marker_coordinates(self, color="red"):
        """
        Uses YOLO to find the specific colored marker on the platform.
        Returns (x, y) coordinates.
        """
        # img = self.vision.capture_frame()
        # detections = self.vision.infer(img)
        # for det in detections:
        #     if det.class_name == color:
        #         return det.x, det.y
        
        # Mock coordinates for demonstration
        if color == "red": return 40, 30
        if color == "blue": return -40, -30
        if color == "green": return -40, 30
        if color == "yellow": return 40, -30
        return 0, 0
        
    def run_sequence(self):
        targets = ["center", "red", "blue", "green", "yellow"]
        HOLD_TIME_MS = 10000 # 10 seconds per target
        
        print("Starting Expert PID Evaluation Sequence...")
        
        for target_color in targets:
            tx, ty = self.get_marker_coordinates(target_color)
            print(f"Moving to {target_color} at ({tx}, {ty}). Holding for 10 seconds...")
            
            start_time = time.time() * 1000
            
            while (time.time() * 1000) - start_time < HOLD_TIME_MS:
                now_ms = time.time() * 1000
                
                # 1. Send Target to MLVisionControl Firmware (which runs the PID)
                # payload = struct.pack('<chh', b'<', int(tx), int(ty))
                # self.serial.write(payload)
                
                # 2. Read Telemetry back from Arduino
                # touch_x, touch_y, theta_a, theta_b, theta_c = self.serial.read_telemetry()
                
                # Mock telemetry reading
                touch_x, touch_y = tx + 5.0, ty - 2.0 # Simulate some steady state error
                theta_a, theta_b, theta_c = 15.0, -10.0, 5.0 # Mock angles
                
                # 3. Log to CSV
                self.writer.writerow([now_ms, tx, ty, touch_x, touch_y, theta_a, theta_b, theta_c])
                
                # Run loop at ~30Hz
                time.sleep(1/30.0)
                
        print("Sequence complete. Data saved to labels_sequential_expert.csv!")
        self.csv_file.close()

if __name__ == "__main__":
    evaluator = ExpertEvaluator()
    evaluator.run_sequence()
