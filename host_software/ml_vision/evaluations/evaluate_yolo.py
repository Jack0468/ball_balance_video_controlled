import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ultralytics import YOLO

def main():
    # Resolve absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the best trained YOLO model weights
    model_path = os.path.abspath(os.path.join(script_dir, "../../../runs/pose/models/new_platform_pose_model/weights/best.pt"))
    
    # Path to the dataset configuration file we want to evaluate on
    data_path = os.path.abspath(os.path.join(script_dir, "../data_processing/raw_dataset.yaml"))
    
    print(f"Evaluating model: {model_path}")
    print(f"On dataset: {data_path}")
    
    # Load model
    model = YOLO(model_path)
    
    # Run evaluation
    # This will automatically print the precision/recall tables to standard out,
    # and save detailed plots, confusion matrices, and metrics to yolo_eval_results/val*/
    eval_project_dir = os.path.abspath(os.path.join(script_dir, "yolo_eval_results"))
    metrics = model.val(data=data_path, imgsz=640, project=eval_project_dir, name="val")

if __name__ == "__main__":
    main()
