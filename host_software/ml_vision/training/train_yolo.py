import os
import argparse
import yaml
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on the auto-generated ball and marker dataset")
    parser.add_argument("--data_dir", required=True, help="Path to the directory containing images and labels")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--save_dir", default="../models", help="Project directory to save the trained YOLO models")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint (.pt) to resume training from")
    args = parser.parse_args()
    
    dataset_yaml_path = os.path.join(args.data_dir, "dataset.yaml")
    
    if not os.path.exists(dataset_yaml_path):
        print(f"ERROR: {dataset_yaml_path} not found. Ensure you are pointing to the synthetic dataset directory.")
        return
    
    # 2. Load a pre-trained YOLOv8-Pose nano model or resume
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming YOLOv8 training from {args.resume}...")
        model = YOLO(args.resume)
    else:
        print("Loading pre-trained YOLOv8n-Pose model...")
        model = YOLO("yolov8n-pose.pt")
    
    # 3. Train the model
    print(f"Starting training for {args.epochs} epochs...")
    
    import torch
    device_str = "0" if torch.cuda.is_available() else "cpu"
    
    # Use workers=0 for Windows to avoid multiprocessing issues
    resume_flag = bool(args.resume and os.path.exists(args.resume))
    
    results = model.train(
        data=dataset_yaml_path,
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=640,
        workers=0,
        project=args.save_dir,
        name="unified_pose_model",
        device=device_str,
        fliplr=0.0,  # Disable horizontal flip to preserve Top-Left/Top-Right semantic order
        flipud=0.0,   # Disable vertical flip
        resume=resume_flag
    )
    
    print("Training complete! Model saved in ../models/unified_pose_model/weights/best.pt")

if __name__ == '__main__':
    main()
