import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from ultralytics import YOLO

def main():
    # We start with the nano model for speed (yolov8n-pose.pt)
    # The model will be automatically downloaded by Ultralytics if not present.
    print("Initializing YOLOv8n-Pose model...")
    model = YOLO('yolov8n-pose.pt')

    # Path to the dataset configuration file we just created
    yaml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/platform_pose.yaml'))

    print(f"Starting training using config: {yaml_path}")
    
    # Train the model!
    # - epochs=50 is a good starting point for 1,000 synthetic images.
    # - imgsz=640 is standard for YOLOv8.
    # - device=0 uses the first available GPU (CUDA). Change to 'cpu' if no GPU is available.
    results = model.train(
        data=yaml_path,
        epochs=50,
        imgsz=640,
        batch=16,
        project=os.path.abspath(os.path.join(os.path.dirname(__file__), '../models')),
        name='platform_pose_model'
    )
    
    print("\nTraining complete!")
    print("The best model weights are saved at: ../models/platform_pose_model/weights/best.pt")

if __name__ == '__main__':
    main()
