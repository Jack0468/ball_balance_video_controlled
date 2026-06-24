"""
ml_model_benchmarks.py

Provides a unified interface for evaluating YOLO and MobileNet SSD models 
against the Classical CV baseline on the synced labeled dataset.
"""

import time
import numpy as np

# Note: Actual YOLO and TFLite implementations will require installing 
# `ultralytics` and `tensorflow` respectively. This file provides the skeleton
# for loading and timing these models.

class YOLOBenchmark:
    def __init__(self, model_path='yolov8n.pt'):
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.available = True
        except ImportError:
            print("ultralytics not installed. YOLO unavailable.")
            self.available = False

    def predict(self, frame):
        if not self.available:
            return None
            
        results = self.model(frame, verbose=False)
        # Assuming the ball is the most confident detection
        for r in results:
            boxes = r.boxes
            if len(boxes) > 0:
                # get center of the first box
                x1, y1, x2, y2 = boxes[0].xyxy[0].cpu().numpy()
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                return (center_x, center_y)
        return None

class Benchmarker:
    def __init__(self):
        self.models = {}
        
    def add_model(self, name, model_instance):
        self.models[name] = model_instance
        
    def run_benchmark(self, frames, labels):
        """
        frames: list of cv2 image matrices
        labels: list of (x, y) ground truth tuples
        """
        results = {}
        for name, model in self.models.items():
            print(f"Benchmarking {name}...")
            start_time = time.time()
            
            mse_list = []
            valid_predictions = 0
            
            for frame, true_pos in zip(frames, labels):
                pred = model.predict(frame)
                if pred is not None and true_pos is not None and not np.isnan(true_pos[0]):
                    err_x = pred[0] - true_pos[0]
                    err_y = pred[1] - true_pos[1]
                    mse = (err_x**2 + err_y**2)
                    mse_list.append(mse)
                    valid_predictions += 1
            
            end_time = time.time()
            fps = len(frames) / (end_time - start_time)
            avg_mse = np.mean(mse_list) if mse_list else float('inf')
            
            results[name] = {
                'fps': fps,
                'avg_mse': avg_mse,
                'valid_predictions': valid_predictions,
                'total_frames': len(frames)
            }
            
        return results

if __name__ == "__main__":
    # Example usage:
    # from classical_cv_model import ClassicalCVModel
    # benchmarker = Benchmarker()
    # benchmarker.add_model("ClassicalCV", ClassicalCVModel())
    # benchmarker.add_model("YOLOv8n", YOLOBenchmark())
    # results = benchmarker.run_benchmark(sample_frames, sample_labels)
    # print(results)
    pass
