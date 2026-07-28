from ultralytics import YOLO
import argparse
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
    model_path = os.path.join(repo_root, 'yolov8n-pose.pt')
    yaml_path = os.path.join(repo_root, 'host_software', 'ml_vision', 'data_processing', 'raw_dataset.yaml')
    project_dir = os.path.join(repo_root, 'models')
    
    # Starting from yolov8n-pose
    print(f"Loading {model_path}...")
    model = YOLO(model_path)
    
    print(f"Starting robust YOLO-Pose training on {yaml_path} for Platform + Markers...")
    
    # Using extreme augmentations to ensure perspective invariance and lighting invariance
    results = model.train(
        data=yaml_path,
        epochs=args.epochs,
        imgsz=640,
        batch=16,
        project='models',
        name='yolov8_platform_pose_markers_v3',
        exist_ok=True,
        # Heavy augmentations
        perspective=0.001, # Perspective warp
        fliplr=0.5,
        degrees=90.0,      # Rotations (increased for rotational invariance)
        translate=0.2,     # Horizontal/vertical shift to improve off-center platform center predictions
        scale=0.9,         # Zoom out/in by 90% (increased for distance invariance)
        mosaic=1.0,        # High mosaic for background variety
        hsv_h=0.015,       # Color jitter (Hue)
        hsv_s=0.7,         # Color jitter (Sat)
        hsv_v=0.4          # Color jitter (Val)
    )
    
    print("Training complete! Model saved in models/yolov8_platform_pose_markers_v3/weights/best.pt")

if __name__ == '__main__':
    main()
