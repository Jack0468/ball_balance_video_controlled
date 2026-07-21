import os
from ultralytics import YOLO

def main():
    # Resolve absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the best trained YOLO model weights
    model_path = os.path.abspath(os.path.join(script_dir, "../../runs/pose/models/new_platform_pose_model/weights/best.pt"))
    
    # Path to the dataset configuration file we want to evaluate on
    data_path = os.path.abspath(os.path.join(script_dir, "../data_processing/raw_dataset.yaml"))
    
    print(f"Evaluating model: {model_path}")
    print(f"On dataset: {data_path}")
    
    # Load model
    model = YOLO(model_path)
    
    # Run evaluation
    # This will automatically print the precision/recall tables to standard out,
    # and save detailed plots, confusion matrices, and metrics to runs/pose/val*/
    metrics = model.val(data=data_path, imgsz=640)

if __name__ == "__main__":
    main()
