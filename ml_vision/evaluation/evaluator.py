"""
evaluator.py

Utility for visualizing the predicted (x, y) coordinates overlayed on the 
original frames and calculating evaluation metrics.
"""

import cv2
import numpy as np
import os

class Evaluator:
    def __init__(self, output_dir="evaluation_results"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def visualize_predictions(self, frame, true_pos, pred_pos, frame_idx):
        """
        Overlays true and predicted positions on the frame and saves it.
        true_pos: (x, y) tuple or None
        pred_pos: (x, y) tuple or None
        """
        viz_frame = frame.copy()
        
        # Draw True Position (Green)
        if true_pos is not None and not np.isnan(true_pos[0]):
            cv2.circle(viz_frame, (int(true_pos[0]), int(true_pos[1])), 8, (0, 255, 0), -1)
            cv2.putText(viz_frame, "True", (int(true_pos[0]) + 10, int(true_pos[1])), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        # Draw Predicted Position (Red)
        if pred_pos is not None:
            cv2.circle(viz_frame, (int(pred_pos[0]), int(pred_pos[1])), 6, (0, 0, 255), -1)
            cv2.putText(viz_frame, "Pred", (int(pred_pos[0]) + 10, int(pred_pos[1]) - 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
        # Draw error line
        if true_pos is not None and pred_pos is not None and not np.isnan(true_pos[0]):
            cv2.line(viz_frame, (int(true_pos[0]), int(true_pos[1])), 
                     (int(pred_pos[0]), int(pred_pos[1])), (255, 0, 0), 1)
                     
        output_path = os.path.join(self.output_dir, f"eval_frame_{frame_idx:05d}.jpg")
        cv2.imwrite(output_path, viz_frame)
        return viz_frame

    def calculate_metrics(self, results_dict):
        """
        Prints and formats the benchmark results.
        """
        print("\n--- Evaluation Metrics ---")
        for model_name, metrics in results_dict.items():
            print(f"Model: {model_name}")
            print(f"  FPS: {metrics['fps']:.2f}")
            print(f"  Average MSE: {metrics['avg_mse']:.2f} pixels^2")
            print(f"  Valid Predictions: {metrics['valid_predictions']} / {metrics['total_frames']}")
            print("-" * 25)
