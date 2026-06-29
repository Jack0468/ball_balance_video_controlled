from ultralytics import YOLO
import os

def export_model(model_path, is_local_path=True):
    print(f"Loading model: {model_path}...")
    
    # Resolve absolute path if it's a local file we know the relative location of
    if is_local_path:
        resolved_path = os.path.abspath(os.path.join(os.path.dirname(__file__), model_path))
    else:
        resolved_path = model_path
        
    model = YOLO(resolved_path)
    
    print(f"Exporting to ONNX format...")
    # dynamic=False is better for TensorRT/FPGA compilers. 
    # opset=12 is widely supported across ONNX Runtime and embedded devices.
    output_path = model.export(format='onnx', opset=12, dynamic=False)
    
    print(f"Success! ONNX model saved to: {output_path}\n")

def main():
    print("--- YOLO to ONNX Exporter ---\n")
    
    # 1. Export the standard YOLOv8n Ball Tracker
    # It will automatically download yolov8n.pt if it isn't in the root dir
    export_model('yolov8n.pt', is_local_path=False)
    
    # 2. Export our custom YOLO-Pose model
    export_model('../models/platform_pose_model/weights/best.pt', is_local_path=True)
    
    print("All models exported successfully! They are ready to be used with onnxruntime or TensorRT.")

if __name__ == "__main__":
    main()
