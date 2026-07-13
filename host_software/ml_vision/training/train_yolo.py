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
    
    # 1. Create a dataset.yaml file dynamically
    dataset_yaml_path = os.path.join(args.data_dir, "dataset.yaml")
    
    yaml_content = {
        "path": os.path.abspath(args.data_dir),
        "train": ".",  # Since our generator just dumped them in one dir for now
        "val": ".",    # We evaluate on the same for this initial script (we can split later)
        "names": {
            0: "ball",
            1: "marker"
        }
    }
    
    with open(dataset_yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
        
    print(f"Created YOLO dataset configuration at {dataset_yaml_path}")
    
    # 2. Load a pre-trained YOLOv8 nano model (fastest inference for edge)
    print("Loading pre-trained YOLOv8n model...")
    model = YOLO("yolov8n.pt")  # It will download automatically if not present
    
    # 3. Train the model
    print(f"Starting training for {args.epochs} epochs...")
    
    # Use workers=0 for Windows to avoid multiprocessing issues
    results = model.train(
        data=dataset_yaml_path,
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=640,
        workers=0,
        project="../models",
        name="yolov8_marker_and_ball_detector",
        device="cpu"  # Assuming we are running on CPU for now; change to 'cuda:0' if GPU available
    )
    
    print("Training complete! Model saved in ../models/yolov8_marker_and_ball_detector/weights/best.pt")

if __name__ == '__main__':
    main()
