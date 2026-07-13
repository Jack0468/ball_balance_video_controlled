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
    def __init__(self, model_path=None):
        if model_path is None:
            import os
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights', 'yolov8n.pt')
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
    import cv2
    import os
    import glob

    # Dynamically find the image directory based on the medallion file structure
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    image_dir = os.path.join(base_dir, "data", "image", "YT_video1")
    image_paths = glob.glob(os.path.join(image_dir, "*.png")) + glob.glob(os.path.join(image_dir, "*.jpg"))
    
    sample_frames = []
    print(f"Loading {len(image_paths)} images from {image_dir}...")
    for path in image_paths[:100]: # Load at most 100 frames to keep the benchmark fast
        frame = cv2.imread(path)
        if frame is not None:
            sample_frames.append(frame)
            
    if not sample_frames:
        print("No images found. Please ensure images exist in the data/image/YT_video1/ folder.")
    else:
        sample_labels = [None] * len(sample_frames)  # Dummy labels since we don't have annotated (Gold) data yet
        
        from classical_cv_model import ClassicalCVModel
        benchmarker = Benchmarker()
        benchmarker.add_model("ClassicalCV", ClassicalCVModel())
        benchmarker.add_model("YOLOv8n", YOLOBenchmark())
        
        results = benchmarker.run_benchmark(sample_frames, sample_labels)
        for name, res in results.items():
            print(f"\n--- {name} Results ---")
            print(f"FPS: {res['fps']:.2f}")
            if res['avg_mse'] == float('inf'):
                print("Avg MSE: N/A (No ground truth labels provided)")
            else:
                print(f"Avg MSE: {res['avg_mse']}")
            print(f"Valid Predictions: {res['valid_predictions']}/{res['total_frames']}")
