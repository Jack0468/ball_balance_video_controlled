from ultralytics import YOLO
import os
import sys
import subprocess

def main():
    print("--- YOLO to OpenVINO Exporter (Intel CPU Optimized) ---\n")
    
    pose_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/platform_pose_model/weights/best.pt'))
    
    print(f"Loading Pose Model: {pose_model_path}...")
    model = YOLO(pose_model_path)
    
    print("Step 1/2: Exporting to ONNX format at 320x320 for ultra-low latency...")
    # imgsz=320 reduces the pixel area by 4x compared to 640x640, massively speeding up CPU inference.
    onnx_path = model.export(format='onnx', imgsz=320, dynamic=False)
    
    print("\nStep 2/2: Converting ONNX to OpenVINO using ovc CLI (bypassing DLL conflicts)...")
    openvino_dir = os.path.join(os.path.dirname(onnx_path), "best_openvino_model")
    
    ovc_exe = os.path.join(os.path.dirname(sys.executable), "Scripts", "ovc.exe")
    
    # Run the ovc CLI to generate the .xml and .bin
    subprocess.run([ovc_exe, onnx_path, "--output_model", openvino_dir], check=True)
    
    print(f"\nSuccess! OpenVINO model saved to: {openvino_dir}")

if __name__ == "__main__":
    main()
