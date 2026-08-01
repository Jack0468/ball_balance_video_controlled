import cv2
import time
import numpy as np
import os
import torch
import serial
import argparse
from ml_vision.core.coordinate_math import HomographyProjector
from ml_audio.audio_receiver_pytorch import AudioCommandReceiver

from src.receivers import USBReceiver, UDPReceiver
from src.utils import find_stm32_port
from src.models import (
    load_yolo_model,
    load_mlp_corrector_v1_model,
    process_vision_frame,
)
from src.state_machine import TargetStateMachine
from src.latency_monitor import RealtimeLatencyMonitor

# --- Configuration ---
SERIAL_PORT = "COM7"
SERIAL_BAUD = 2000000
# ---------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam_id", type=int, default=0, help="Camera ID for USB mode")
    parser.add_argument(
        "--port",
        type=str,
        default="auto",
        help="STM32 serial port (e.g. COM3 or 'auto')",
    )
    parser.add_argument(
        "--udp", action="store_true", help="Use UDP receiver instead of USB camera"
    )
    parser.add_argument(
        "--udp_port",
        type=int,
        default=5001,
        help="Port to listen on for UDP video stream",
    )
    args = parser.parse_args()

    # Auto-detect port if 'auto'
    if args.port == "auto":
        detected_port = find_stm32_port()
        if detected_port:
            print(f"Auto-detected STM32 on {detected_port}")
            args.port = detected_port
        else:
            args.port = SERIAL_PORT
            print(f"Could not auto-detect STM32. Defaulting to {args.port}")

    # 1. Hardware/Model Init
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    yolo_path = os.path.abspath(
        os.path.join(
            script_dir,
            "ml_vision/models/yolov8_platform_pose_markers_v4/weights/best.pt",
        )
    )
    mlp_corrector_path = os.path.abspath(
        os.path.join(
            script_dir, "ml_vision/models/mlp_corrector_v6/best_mlp_corrector_v6.pth"
        )
    )

    yolo_model = load_yolo_model(yolo_path, device)
    mlp_corrector_v1_model = load_mlp_corrector_v1_model(mlp_corrector_path, device)

    # Initialize Homography Projector
    dst_pts = np.array([[-70, 55], [70, 55], [70, -55], [-70, -55]], dtype=np.float32)
    projector = HomographyProjector(dst_pts)

    # 2. Audio & State Init
    audio_model_path = os.path.abspath(
        os.path.join(
            script_dir,
            "ml_audio/models/pytorch_v3/audio_command_classifier_state_dict_v3.pth",
        )
    )
    audio_receiver = AudioCommandReceiver(audio_model_path)
    state_machine = TargetStateMachine()

    latency_monitor = RealtimeLatencyMonitor(
        log_interval=100, save_dir=os.path.join(script_dir, "ml_vision/evaluations")
    )

    # 3. Serial Port Init
    try:
        ser = serial.Serial(args.port, SERIAL_BAUD, timeout=0)
        print(f"Connected to STM32 on {args.port} at {SERIAL_BAUD} baud.")
    except Exception as e:
        print(
            f"Could not open serial port {args.port}. Continuing in dry-run mode (no serial transmission)."
        )
        ser = None

    # 3. Camera/Receiver Init
    if args.udp:
        receiver = UDPReceiver(port=args.udp_port, width=640, height=480)
    else:
        receiver = USBReceiver(camera_id=args.cam_id)

    # Wait for the first frame
    print("Waiting for camera feed...")
    frame = None
    while frame is None:
        frame = receiver.get_latest_frame()
        time.sleep(0.1)

    print(f"Starting YOLO Main Inference Loop... (Headless mode, press Ctrl+C to quit)")
    try:
        while True:
            frame = receiver.get_latest_frame()
            if frame is None:
                continue

            start_t = time.perf_counter()
            latency_monitor.start_frame()

            # Inference Phase
            cam_x, cam_y, marker_coords = process_vision_frame(
                frame, yolo_model, mlp_corrector_v1_model, projector, device
            )
            yolo_t = time.perf_counter()
            latency_monitor.end_vision()

            if cam_x is None:
                end_t = time.perf_counter()
                fps = 1.0 / (end_t - start_t)
                print(f"Missing detections - Ball/Platform not found | FPS: {fps:.1f}")
                continue
            mlp_t = time.perf_counter()
            latency_monitor.end_mlp()

            # Process Audio Commands
            command = audio_receiver.get_latest_command()
            if command:
                print(f"\n[AUDIO] Heard command: {command}\n")
            state_machine.process_command(command, cam_x, cam_y)
            state_machine.update_markers(marker_coords)
            target_x, target_y = state_machine.get_target_coords()
            latency_monitor.end_audio()

            # Serial Transmission Phase
            # We send cam_x, cam_y, target_x, target_y so the RL policy gets true absolute coords
            # and target coords, keeping tilt dynamics perfectly aligned.
            try:
                payload = (
                    f"{cam_x:.2f},{cam_y:.2f},{target_x:.2f},{target_y:.2f}\n".encode(
                        "ascii"
                    )
                )
                if ser is not None:
                    ser.write(payload)
            except Exception as e:
                print(f"Serial Error: {e}")

            end_t = time.perf_counter()
            latency_monitor.end_frame(log_to_console=False)

            total_latency_ms = (end_t - start_t) * 1000.0
            yolo_latency_ms = (yolo_t - start_t) * 1000.0
            mlp_latency_ms = (mlp_t - yolo_t) * 1000.0
            fps = 1.0 / (end_t - start_t)

            marker_str = ", ".join(
                [f"{name}=({x:.1f},{y:.1f})" for name, (x, y) in marker_coords.items()]
            )
            marker_out = f" | Markers: {marker_str}" if marker_str else ""

            print(
                f"Targeting '{state_machine.current_target_name}' at X={target_x:.1f} Y={target_y:.1f} | Ball: X={cam_x:.1f} Y={cam_y:.1f} mm | FPS: {fps:.1f} {marker_out}"
            )

    except KeyboardInterrupt:
        pass
    finally:
        receiver.stop()
        audio_receiver.stop()
        if ser:
            ser.close()
        cv2.destroyAllWindows()
        print("Inference loop stopped.")


if __name__ == "__main__":
    main()
