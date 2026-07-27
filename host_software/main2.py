import cv2
import threading
import queue
import time
import struct
import numpy as np
import os
import torch
import torch.nn as nn
import serial
import argparse
from datetime import datetime
from ml_vision.core.coordinate_math import HomographyProjector
from ml_audio.audio_receiver_pytorch import AudioCommandReceiver

from src.receivers import USBReceiver, UDPReceiver
from src.utils import find_stm32_port
from src.models import load_yolo_model, load_mlp_corrector_v1_model, process_vision_frame
from src.state_machine import TargetStateMachine
from src.latency_monitor import RealtimeLatencyMonitor
from src.touch_logger import TouchTelemetryLogger

# --- Configuration ---
SERIAL_PORT = "COM7"
SERIAL_BAUD = 2000000
# ---------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam_id", type=int, default=0, help="Camera ID for USB mode")
    parser.add_argument("--port", type=str, default="auto", help="STM32 serial port (e.g. COM3 or 'auto')")
    parser.add_argument("--udp", action="store_true", help="Use UDP receiver instead of USB camera")
    parser.add_argument("--udp_port", type=int, default=5001, help="Port to listen on for UDP video stream")
    # CHANGED: ground-truth logging controls.
    parser.add_argument("--log_csv", type=str, default="auto",
                        help="Path for the vision-vs-touchscreen CSV. "
                             "'auto' timestamps a file under ml_vision/evaluations/. "
                             "'off' disables logging.")
    parser.add_argument("--quiet_mcu", action="store_true",
                        help="Suppress '#' status lines coming back from the MCU")
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
    yolo_path = os.path.abspath(os.path.join(script_dir, 'ml_vision/models/yolov8_platform_pose_markers_v1/weights/best.pt'))
    mlp_corrector_v1_path = os.path.abspath(os.path.join(script_dir, 'ml_vision/models/mlp_corrector_v1/best_corrector.pth'))

    yolo_model = load_yolo_model(yolo_path, device)
    mlp_corrector_v1_model = load_mlp_corrector_v1_model(mlp_corrector_v1_path, device)

    # Initialize Homography Projector
    # NOTE: these bounds (+/-70, +/-55 mm) define the VISION frame. The
    # touchscreen maps its glass to +/-93.75 x +/-70.5 mm. Reconcile the two
    # before you read anything into err_mm -- see the FRAME MATCH block in
    # TouchProbe.cpp.
    dst_pts = np.array([
        [-70, 55],
        [70, 55],
        [70, -55],
        [-70, -55]
    ], dtype=np.float32)
    projector = HomographyProjector(dst_pts)

    # 2. Audio & State Init
    audio_model_path = os.path.abspath(os.path.join(script_dir, 'ml_audio/audio_command_classifier_state_dict_v2.pth'))
    audio_receiver = AudioCommandReceiver(audio_model_path)
    state_machine = TargetStateMachine()

    latency_monitor = RealtimeLatencyMonitor(log_interval=100, save_dir=os.path.join(script_dir, 'ml_vision/evaluations'))

    # 3. Serial Port Init
    # CHANGED: timeout 0 -> 0.02. The link is bidirectional now; a small read
    # timeout lets the reader thread block briefly instead of spinning the CPU.
    # Writes are unaffected (that is write_timeout, which we leave unset).
    try:
        ser = serial.Serial(args.port, SERIAL_BAUD, timeout=0.02)
        print(f"Connected to STM32 on {args.port} at {SERIAL_BAUD} baud.")
    except Exception as e:
        print(f"Could not open serial port {args.port}. Continuing in dry-run mode (no serial transmission).")
        ser = None

    # 3b. NEW: touchscreen ground-truth logger (reader thread + CSV).
    touch_logger = None
    if args.log_csv != "off" and ser is not None:
        if args.log_csv == "auto":
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = os.path.join(script_dir, 'ml_vision/evaluations',
                                    f'ground_truth_{stamp}.csv')
        else:
            csv_path = args.log_csv
        touch_logger = TouchTelemetryLogger(
            ser, csv_path, print_status_lines=not args.quiet_mcu)
        touch_logger.start()

    # 4. Camera/Receiver Init
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

    seq = 0  # NEW: frame counter, echoed back by the MCU to join the two streams

    print(f"Starting YOLO Main Inference Loop... (Headless mode, press Ctrl+C to quit)")
    try:
        while True:
            frame = receiver.get_latest_frame()
            if frame is None:
                continue

            start_t = time.perf_counter()
            latency_monitor.start_frame()

            # Inference Phase
            cam_x, cam_y, marker_coords = process_vision_frame(frame, yolo_model, mlp_corrector_v1_model, projector, device)
            yolo_t = time.perf_counter()
            latency_monitor.end_vision()

            if cam_x is None:
                # CHANGED: tell the MCU we lost the ball instead of going silent.
                # Silence used to be indistinguishable from a stalled host; now
                # the firmware's 150 ms staleness timer and an explicit "lost"
                # are two different things in the log.
                if ser is not None:
                    try:
                        ser.write(b"L\n")
                    except Exception as e:
                        print(f"Serial Error: {e}")
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
            # CHANGED: prefix "V,<seq>". Same numbers as before, plus a join key
            # so each telemetry record can be matched to the exact vision frame
            # that produced it. The firmware still accepts the old untagged form.
            seq += 1
            try:
                payload = f"V,{seq},{cam_x:.2f},{cam_y:.2f},{target_x:.2f},{target_y:.2f}\n".encode('ascii')
                send_ts = time.perf_counter()
                if ser is not None:
                    ser.write(payload)
                if touch_logger is not None:
                    touch_logger.register_frame(seq, cam_x, cam_y,
                                                target_x, target_y, send_ts)
            except Exception as e:
                print(f"Serial Error: {e}")

            end_t = time.perf_counter()
            latency_monitor.end_frame(log_to_console=False)

            total_latency_ms = (end_t - start_t) * 1000.0
            yolo_latency_ms = (yolo_t - start_t) * 1000.0
            mlp_latency_ms = (mlp_t - yolo_t) * 1000.0
            fps = 1.0 / (end_t - start_t)

            marker_str = ", ".join([f"{name}=({x:.1f},{y:.1f})" for name, (x, y) in marker_coords.items()])
            marker_out = f" | Markers: {marker_str}" if marker_str else ""

            # NEW: live error readout so you can see calibration drift without
            # opening the CSV.
            gt_out = ""
            if touch_logger is not None and touch_logger.last_err_mm is not None:
                gt_out = f" | GT err: {touch_logger.last_err_mm:.1f}mm"

            print(f"Targeting '{state_machine.current_target_name}' at X={target_x:.1f} Y={target_y:.1f} | Ball: X={cam_x:.1f} Y={cam_y:.1f} mm | FPS: {fps:.1f}{gt_out}{marker_out}")

    except KeyboardInterrupt:
        pass
    finally:
        receiver.stop()
        audio_receiver.stop()
        # Stop the reader before closing the port, or the thread will raise on a
        # closed handle.
        if touch_logger is not None:
            touch_logger.stop()
        if ser:
            ser.close()
        cv2.destroyAllWindows()
        print("Inference loop stopped.")

if __name__ == '__main__':
    main()