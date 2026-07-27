import cv2
import time
import struct
import numpy as np
import os
import torch
import serial
import argparse
import csv

from ml_vision.core.coordinate_math import HomographyProjector
from ml_audio.audio_receiver_pytorch import AudioCommandReceiver

from src.receivers import USBReceiver
from src.utils import find_stm32_port
from src.models import load_yolo_model, load_mlp_corrector_v1_model
from src.state_machine import TargetStateMachine

# --- Configuration ---
SERIAL_PORT = "COM3"
SERIAL_BAUD = 2000000 
EVAL_DURATION = 120 # Total seconds
# ---------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam_id", type=int, default=0, help="Camera ID")
    parser.add_argument("--port", type=str, default="auto", help="STM32 serial port")
    args = parser.parse_args()

    if args.port == "auto":
        detected_port = find_stm32_port()
        args.port = detected_port if detected_port else SERIAL_PORT

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Paths
    yolo_path = os.path.abspath(os.path.join(script_dir, 'ml_vision/models/yolov8_platform_markers_v1/weights/best.pt'))
    mlp_corrector_v1_path = os.path.abspath(os.path.join(script_dir, 'ml_vision/models/mlp_corrector_v1/best_mlp_corrector_v1.pth'))
    audio_model_path = os.path.abspath(os.path.join(script_dir, 'ml_audio/synthetic/models/pytorch/audio_weights_with_synthetic.pth'))
    master_audio_path = os.path.abspath(os.path.join(script_dir, 'ml_audio', 'data', '02_silver', 'master_evaluation_audio.wav'))
    
    if not os.path.exists(master_audio_path):
        print(f"ERROR: {master_audio_path} not found!")
        print("Please run `python ml_audio/create_master_audio.py` first to generate the streaming sequence.")
        return

    # Output CSV
    out_dir = os.path.abspath(os.path.join(script_dir, 'data', '04_evaluation'))
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'expert_evaluation_run.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["host_timestamp_ms", "mcu_micros", "target_x", "target_y", 
                         "touch_x", "touch_y", "error_x", "error_y", "pitch", "roll", 
                         "theta_a", "theta_b", "theta_c", "integral_x", "integral_y", 
                         "deriv_x", "deriv_y"])

    # Load Models
    yolo_model = load_yolo_model(yolo_path, device)
    mlp_corrector_v1_model = load_mlp_corrector_v1_model(mlp_corrector_v1_path, device)
    
    dst_pts = np.array([[-70, 55], [70, 55], [70, -55], [-70, -55]], dtype=np.float32)
    projector = HomographyProjector(dst_pts)
    state_machine = TargetStateMachine()
    
    try:
        ser = serial.Serial(args.port, SERIAL_BAUD, timeout=0.01)
        print(f"Connected to STM32 on {args.port} at {SERIAL_BAUD} baud.")
    except Exception as e:
        print(f"Serial port {args.port} unavailable. Evaluation requires physical hardware!")
        return

    receiver = USBReceiver(camera_id=args.cam_id)
    
    print("\nWaiting for camera feed...")
    while receiver.get_latest_frame() is None:
        time.sleep(0.1)
        
    print("\nWaiting for ball to be placed and balanced in the center...")
    balanced_start_time = None
    
    while True:
        frame = receiver.get_latest_frame()
        if frame is None:
            continue
            
        results = yolo_model.predict(source=frame, imgsz=320, conf=0.5, verbose=False)
        if not results or len(results) == 0 or results[0].boxes is None:
            balanced_start_time = None
            continue
            
        res = results[0]
        classes = res.boxes.cls.cpu().numpy()
        boxes = res.boxes.xywh.cpu().numpy()
        
        corners = None
        ball_box = None
        
        for i, c_id in enumerate(classes):
            c = int(c_id)
            if c == 0:
                if res.keypoints is not None and len(res.keypoints.xy) > i:
                    kpts = res.keypoints.xy[i].cpu().numpy()
                    if len(kpts) == 4: corners = kpts
            elif c == 1:
                ball_box = boxes[i]
                
        if corners is None or ball_box is None:
            balanced_start_time = None
            continue
            
        homography_x, homography_y = 0.0, 0.0
        if projector.update_homography(corners):
            hx, hy = projector.project_point(ball_box[0], ball_box[1])
            if hx is not None and hy is not None:
                homography_x, homography_y = hx, hy
                    
        features = np.array([
            ball_box[0], ball_box[1], ball_box[2], ball_box[3],
            corners[0][0], corners[0][1], corners[1][0], corners[1][1],
            corners[2][0], corners[2][1], corners[3][0], corners[3][1],
            homography_x, homography_y
        ], dtype=np.float32)
        features[0:12:2] /= 640.0
        features[1:12:2] /= 480.0
        features[12:] /= 100.0
        input_tensor = torch.tensor(features).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = mlp_corrector_v1_model(input_tensor)
        cam_x, cam_y = output[0].cpu().numpy()
        
        # Send center target to STM32 (ASCII)
        payload = f"{cam_x:.2f},{cam_y:.2f},0.0,0.0\n".encode('ascii')
        ser.write(payload)
        
        # Check if balanced (within 20mm radius)
        dist = np.sqrt(cam_x**2 + cam_y**2)
        if dist < 20.0:
            if balanced_start_time is None:
                balanced_start_time = time.time()
            elif time.time() - balanced_start_time > 2.0:
                print("\nBall successfully balanced in the center!")
                break
        else:
            balanced_start_time = None
            
        # Drain serial buffer so we don't back up while waiting
        while ser.in_waiting > 0:
            ser.read(ser.in_waiting)
    
    # Initialize the audio receiver in FILE STREAMING mode ONLY AFTER ball is placed!
    audio_receiver = AudioCommandReceiver(audio_model_path, source_file=master_audio_path)

    print("\n===========================================")
    print("STARTING TRUE STREAMING EXPERT EVALUATION!")
    print("===========================================\n")
    
    start_time = time.time()
    
    # Binary struct format from ExpertEvaluationFirmware
    struct_format = "<Ifffffffffffffff"
    expected_size = struct.calcsize(struct_format) + 4 # +4 for sync header
    sync_buf = bytearray()
    
    try:
        while True:
            elapsed = time.time() - start_time
            if elapsed > EVAL_DURATION:
                print("Evaluation sequence complete! Stream finished.")
                break
                
            # Pop commands from the streaming audio model exactly like live mode
            command = audio_receiver.get_latest_command()
            if command:
                print(f"[{elapsed:.1f}s] STREAM DETECTED COMMAND: {command}")
                state_machine.process_command(command)

            frame = receiver.get_latest_frame()
            if frame is None:
                continue
                
            # 1. Vision Inference
            results = yolo_model.predict(source=frame, imgsz=320, conf=0.5, verbose=False)
            if not results or len(results) == 0 or results[0].boxes is None:
                continue
                
            res = results[0]
            classes = res.boxes.cls.cpu().numpy()
            boxes = res.boxes.xywh.cpu().numpy()
            
            corners = None
            ball_box = None
            detected_markers = {}
            
            for i, c_id in enumerate(classes):
                c = int(c_id)
                if c == 0:
                    if res.keypoints is not None and len(res.keypoints.xy) > i:
                        kpts = res.keypoints.xy[i].cpu().numpy()
                        if len(kpts) == 4: corners = kpts
                elif c == 1:
                    ball_box = boxes[i]
                elif c >= 2:
                    name = yolo_model.names[c].replace('_marker', '')
                    detected_markers[name] = boxes[i]
                    
            if corners is None or ball_box is None:
                continue
                
            # Homography & Target Extraction
            homography_x, homography_y = 0.0, 0.0
            marker_coords = {}
            if projector.update_homography(corners):
                hx, hy = projector.project_point(ball_box[0], ball_box[1])
                if hx is not None and hy is not None:
                    homography_x, homography_y = hx, hy
                for name, box in detected_markers.items():
                    mx, my = projector.project_point(box[0], box[1])
                    if mx is not None and my is not None:
                        marker_coords[name] = (mx, my)
                        
            # MLP Corrector
            features = np.array([
                ball_box[0], ball_box[1], ball_box[2], ball_box[3],
                corners[0][0], corners[0][1], corners[1][0], corners[1][1],
                corners[2][0], corners[2][1], corners[3][0], corners[3][1],
                homography_x, homography_y
            ], dtype=np.float32)
            features[0:12:2] /= 640.0
            features[1:12:2] /= 480.0
            features[12:] /= 100.0
            input_tensor = torch.tensor(features).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = mlp_corrector_v1_model(input_tensor)
            cam_x, cam_y = output[0].cpu().numpy()
            
            # Target Calculation
            state_machine.update_markers(marker_coords)
            target_x, target_y = state_machine.get_target_coords()
            
            # Send Target to STM32 (ASCII)
            payload = f"{cam_x:.2f},{cam_y:.2f},{target_x:.2f},{target_y:.2f}\n".encode('ascii')
            ser.write(payload)
            
            # 2. Read Binary Telemetry from STM32
            while ser.in_waiting > 0:
                b = ser.read(1)
                sync_buf.append(b[0])
                if len(sync_buf) > 4:
                    sync_buf.pop(0)
                    
                if bytes(sync_buf) == b'\xAA\xBB\xCC\xDD':
                    data = ser.read(expected_size - 4)
                    if len(data) == expected_size - 4:
                        host_time_ms = int(time.time() * 1000)
                        unpacked = struct.unpack(struct_format, data)
                        csv_writer.writerow([host_time_ms] + list(unpacked))
                    sync_buf.clear()
                    
    except KeyboardInterrupt:
        print("\nEvaluation aborted by user.")
    finally:
        audio_receiver.stop()
        receiver.stop()
        ser.close()
        csv_file.close()
        cv2.destroyAllWindows()
        print(f"Data saved to {csv_path}")

if __name__ == '__main__':
    main()
