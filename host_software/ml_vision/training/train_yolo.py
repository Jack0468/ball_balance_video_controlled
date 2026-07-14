import os
import argparse
import yaml
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on the auto-generated ball and marker dataset")
    parser.add_argument("--data_dir", required=True, help="Path to the directory containing images and labels")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    args = parser.parse_args()
    
    dataset_yaml_path = os.path.join(args.data_dir, "dataset.yaml")
    
    if not os.path.exists(dataset_yaml_path):
        print(f"ERROR: {dataset_yaml_path} not found. Ensure you are pointing to the synthetic dataset directory.")
        return
    
    # 2. Load a pre-trained YOLOv8 nano model (fastest inference for edge)
    print("Loading pre-trained YOLOv8n model...")
    model = YOLO("yolov8n.pt")  # It will download automatically if not present
    
    # 3. Train the model
    print(f"Starting training for {args.epochs} epochs...")
    
    import torch
    device_str = "0" if torch.cuda.is_available() else "cpu"
    
    # Use workers=0 for Windows to avoid multiprocessing issues
    results = model.train(
        data=dataset_yaml_path,
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=640,
        workers=0,
        project="../models",
        name="yolov8_marker_and_ball_detector",
        device=device_str
    )
    
    print("Training complete! Model saved in ../models/yolov8_marker_and_ball_detector/weights/best.pt")

if __name__ == '__main__':
    main()
