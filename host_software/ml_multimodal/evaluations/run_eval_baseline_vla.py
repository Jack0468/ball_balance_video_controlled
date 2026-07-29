import time
import csv
import os
import struct
import torch

# from src.openvino_dispatcher import OpenVINODispatcher
# from src.serial_interface import SerialConnection
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ml_multimodal.core.vla_architecture import RT1LiteVLA

class VLAEvaluator:
    def __init__(self, port="COM3", baud=115200, is_baseline=True):
        print(f"Connecting to Arduino VLADirectControl on {port}...")
        # self.serial = SerialConnection(port, baud)
        # self.vision = OpenVINODispatcher("models/yolov8_marker_best.xml")
        
        self.is_baseline = is_baseline
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the VLA model
        self.vla_model = RT1LiteVLA().to(self.device)
        self.vla_model.eval()
        
        self.output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "04_evaluation")
        os.makedirs(self.output_dir, exist_ok=True)
        
        filename = "labels_sequential_baseline_vla.csv" if is_baseline else "labels_sequential_our_vla.csv"
        self.csv_file = open(os.path.join(self.output_dir, filename), "w", newline="")
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow(["host_timestamp_ms", "target_x", "target_y", "touch_x", "touch_y", "theta_a", "theta_b", "theta_c"])
        
        # Audio mapping
        self.vocab = {"hold": 0, "go red": 1, "go blue": 2, "go green": 3, "go yellow": 4}
        
    def get_marker_coordinates(self, color="red"):
        # Mock coordinates
        if color == "red": return 40, 30
        if color == "blue": return -40, -30
        if color == "green": return -40, 30
        if color == "yellow": return 40, -30
        return 0, 0
        
    def run_sequence(self):
        targets = [("center", "hold"), ("red", "go red"), ("blue", "go blue"), ("green", "go green"), ("yellow", "go yellow")]
        HOLD_TIME_MS = 10000 
        
        print("Starting VLA Evaluation Sequence...")
        
        for color, audio_cmd in targets:
            tx, ty = self.get_marker_coordinates(color)
            print(f"Moving to {color} via audio command '{audio_cmd}'. Holding for 10 seconds...")
            
            start_time = time.time() * 1000
            
            while (time.time() * 1000) - start_time < HOLD_TIME_MS:
                now_ms = time.time() * 1000
                
                # Mock current state
                touch_x, touch_y = tx + 8.0, ty + 2.0 
                
                # 1. Run VLA Inference
                # In reality, pass the image from self.vision.capture_frame()
                dummy_img = torch.zeros(1, 3, 224, 224).to(self.device)
                cmd_idx = torch.tensor([self.vocab[audio_cmd]]).to(self.device)
                state = torch.tensor([[touch_x, touch_y]], dtype=torch.float32).to(self.device)
                
                with torch.no_grad():
                    action = self.vla_model(dummy_img, cmd_idx, state)[0]
                    
                theta_a, theta_b, theta_c = action[0].item(), action[1].item(), action[2].item()
                
                # 2. Send Target to VLADirectControl Firmware
                # Scale floats by 100 as per SerialControl.cpp expectations
                # payload = struct.pack('<chhh', b'<', int(theta_a * 100), int(theta_b * 100), int(theta_c * 100))
                # self.serial.write(payload)
                
                # 3. Log to CSV
                self.writer.writerow([now_ms, tx, ty, touch_x, touch_y, theta_a, theta_b, theta_c])
                
                # Run loop at ~30Hz
                time.sleep(1/30.0)
                
        print("Sequence complete. Data saved!")
        self.csv_file.close()

if __name__ == "__main__":
    evaluator = VLAEvaluator(is_baseline=True)
    evaluator.run_sequence()
