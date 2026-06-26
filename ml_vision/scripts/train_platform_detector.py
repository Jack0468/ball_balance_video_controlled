import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from ultralytics import YOLO

def main():
    # 1. Initialize standard YOLOv8n object detection model
    # (Notice we are not using yolov8n-pose.pt)
    model = YOLO('yolov8n.pt')
    
    # 2. Set absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_yaml = os.path.join(script_dir, '../data/platform_bbox.yaml')
    project_dir = os.path.join(script_dir, '../models')
    
    # 3. Train the model
    # Bounding box detection is exponentially easier to learn than keypoint regression.
    # Therefore, we only need 10 epochs for it to converge on our synthetic dataset.
    results = model.train(
        data=data_yaml,
        epochs=1,
        imgsz=160,
        batch=16,
        project=project_dir,
        name='platform_bbox_model',
        device='cpu' # Assuming CPU training is required
    )
    
    print("Training complete! Model saved to:", os.path.join(project_dir, 'platform_bbox_model'))

if __name__ == "__main__":
    main()
