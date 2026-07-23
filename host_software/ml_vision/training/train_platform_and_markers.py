from ultralytics import YOLO
import argparse
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    
    # Starting from yolov8n-pose
    print("Loading yolov8n-pose.pt...")
    model = YOLO("yolov8n-pose.pt")
    
    yaml_path = os.path.abspath('host_software/ml_vision/data_processing/raw_dataset.yaml')
    
    print(f"Starting robust YOLO-Pose training on {yaml_path} for Platform + Markers...")
    
    # Using extreme augmentations to ensure perspective invariance and lighting invariance
    results = model.train(
        data=yaml_path,
        epochs=args.epochs,
        imgsz=640,
        batch=16,
        project='models',
        name='yolo_platform_markers_v2',
        exist_ok=True,
        # Heavy augmentations
        perspective=0.001, # Perspective warp
        degrees=90.0,      # Rotations (increased for rotational invariance)
        scale=0.9,         # Zoom out/in by 90% (increased for distance invariance)
        mosaic=1.0,        # High mosaic for background variety
        hsv_h=0.015,       # Color jitter (Hue)
        hsv_s=0.7,         # Color jitter (Sat)
        hsv_v=0.4          # Color jitter (Val)
    )
    
    print("Training complete! Model saved in models/yolo_platform_markers_v2/weights/best.pt")

if __name__ == '__main__':
    main()
