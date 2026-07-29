import cv2
import time
import numpy as np
import os
import torch
import argparse

import sys
import os
import serial
from torchvision import transforms

# Adjust path to find modules in host_software root
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)
from src.receivers import USBReceiver, UDPReceiver
from src.utils import find_stm32_port
from src.models import load_expert_model, load_yolo_model

# --- Configuration ---
SERIAL_PORT       = "COM7"
SERIAL_BAUD       = 2000000
MAX_BOUND         = 200.0   # ResNet denormalisation constant (must match BallDataset)
CROP_PAD          = 20      # Pixels of padding around the YOLO platform crop
YOLO_POLL_INTERVAL = 1.0    # Seconds between YOLO checks (gate refresh rate)
# ---------------------

def main():
    parser = argparse.ArgumentParser(description="Cascaded YOLO→ResNet ball tracker")
    parser.add_argument("--cam_id",   type=int, default=0,    help="Camera ID for USB mode")
    parser.add_argument("--port",     type=str, default="auto", help="STM32 serial port or 'auto'")
    parser.add_argument("--udp",      action="store_true",    help="Use UDP receiver")
    parser.add_argument("--udp_port", type=int, default=5001, help="UDP listen port")
    args = parser.parse_args()

    # Auto-detect serial port
    if args.port == "auto":
        detected_port = find_stm32_port()
        if detected_port:
            print(f"Auto-detected STM32 on {detected_port}")
            args.port = detected_port
        else:
            args.port = SERIAL_PORT
            print(f"Could not auto-detect STM32. Defaulting to {args.port}")

    # ---- 1. Model Init ----
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    script_dir = root_dir

    yolo_path  = os.path.abspath(os.path.join(script_dir, 'ml_vision/models/yolov8_platform_pose_markers_v4/weights/best.pt'))
    resnet_path = os.path.abspath(os.path.join(script_dir, 'ml_vision/models/resnet18_expert_tracker_v6/expert_tracker_best.pth'))

    yolo_model  = load_yolo_model(yolo_path, device)
    resnet_model = load_expert_model(resnet_path, device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((240, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # ---- 2. Serial Port Init ----
    try:
        ser = serial.Serial(args.port, SERIAL_BAUD, timeout=0)
        print(f"Connected to STM32 on {args.port} at {SERIAL_BAUD} baud.")
    except Exception:
        print(f"Could not open serial port {args.port}. Continuing in dry-run mode.")
        ser = None

    # ---- 3. Camera/Receiver Init ----
    if args.udp:
        receiver = UDPReceiver(port=args.udp_port, width=640, height=480)
    else:
        receiver = USBReceiver(camera_id=args.cam_id)

    print("Waiting for camera feed...")
    frame = None
    while frame is None:
        frame = receiver.get_latest_frame()
        time.sleep(0.1)

    print(f"Starting cascaded YOLO→ResNet loop... (press Ctrl+C to quit)")
    print(f"  Stage 1 (gate): YOLO v3 — polls every {YOLO_POLL_INTERVAL:.1f}s, caches platform+ball presence")
    print(f"  Stage 2 (track): ResNet18 — runs every frame when gate is open")
    print()

    # YOLO gate state — updated at most once per YOLO_POLL_INTERVAL seconds
    yolo_gate_open  = False   # True when platform + ball were last seen
    last_yolo_time  = -YOLO_POLL_INTERVAL  # force a check on the very first frame

    try:
        while True:
            frame = receiver.get_latest_frame()
            if frame is None:
                continue

            start_t = time.perf_counter()
            h_frame, w_frame = frame.shape[:2]

            # ----------------------------------------------------------------
            # Stage 1 — YOLO gate (runs at most once per YOLO_POLL_INTERVAL)
            # ----------------------------------------------------------------
            yolo_ms = 0.0
            now = time.perf_counter()
            if now - last_yolo_time >= YOLO_POLL_INTERVAL:
                yolo_t0 = time.perf_counter()
                results = yolo_model.predict(source=frame, imgsz=320, conf=0.5, verbose=False)
                yolo_ms = (time.perf_counter() - yolo_t0) * 1000.0
                last_yolo_time = time.perf_counter()

                platform_seen = False
                ball_seen     = False

                if results and len(results) > 0 and results[0].boxes is not None:
                    res     = results[0]
                    classes = res.boxes.cls.cpu().numpy()
                    for i, cls in enumerate(classes):
                        c = int(cls)
                        if c == 0:
                            if res.keypoints is not None and len(res.keypoints.xy) > i:
                                kpts = res.keypoints.xy[i].cpu().numpy()
                                if len(kpts) == 4:
                                    platform_seen = True
                        elif c == 1:
                            ball_seen = True

                yolo_gate_open = platform_seen and ball_seen

                if not yolo_gate_open:
                    missing = []
                    if not platform_seen: missing.append("platform")
                    if not ball_seen:     missing.append("ball")
                    print(f"[YOLO] {' + '.join(missing)} not detected — gate closed ({yolo_ms:.1f}ms)")
                else:
                    print(f"[YOLO] Gate open — platform + ball confirmed ({yolo_ms:.1f}ms)")

            if not yolo_gate_open:
                continue

            # ----------------------------------------------------------------
            # Stage 2 — ResNet on full raw frame (gate is open)
            # (Crop to platform bbox is commented out for now)
            # ----------------------------------------------------------------
            # xs = platform_kpts[:, 0]
            # ys = platform_kpts[:, 1]
            # x1 = int(max(0,        xs.min() - CROP_PAD))
            # y1 = int(max(0,        ys.min() - CROP_PAD))
            # x2 = int(min(w_frame,  xs.max() + CROP_PAD))
            # y2 = int(min(h_frame,  ys.max() + CROP_PAD))
            # if x2 <= x1 or y2 <= y1:
            #     print("YOLO: Degenerate crop — skipping")
            #     continue
            # crop = frame[y1:y2, x1:x2]
            # rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            # input_tensor = preprocess(rgb_crop).unsqueeze(0).to(device)

            resnet_t0 = time.perf_counter()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = preprocess(rgb_frame).unsqueeze(0).to(device)

            with torch.no_grad():
                output = resnet_model(input_tensor)

            resnet_ms = (time.perf_counter() - resnet_t0) * 1000.0

            norm_x, norm_y = output[0].cpu().numpy()
            cam_x = float(norm_x * MAX_BOUND)
            cam_y = float(norm_y * MAX_BOUND)

            # ----------------------------------------------------------------
            # Serial Transmission — target is always centre (0, 0)
            # Firmware expects: cam_x, cam_y, target_x, target_y
            # ----------------------------------------------------------------
            try:
                payload = f"{cam_x:.2f},{cam_y:.2f},0.00,0.00\n".encode('ascii')
                if ser is not None:
                    ser.write(payload)
            except Exception as e:
                print(f"Serial Error: {e}")

            end_t     = time.perf_counter()
            total_ms  = (end_t - start_t) * 1000.0
            fps       = 1.0 / (end_t - start_t)

            yolo_str = f", YOLO={yolo_ms:.1f}ms" if yolo_ms > 0 else ""
            print(
                f"Ball: X={cam_x:+6.1f} Y={cam_y:+6.1f} mm | Target: centre (0,0) | "
                f"FPS: {fps:.1f} | Total={total_ms:.1f}ms (ResNet={resnet_ms:.1f}ms{yolo_str})"
            )

    except KeyboardInterrupt:
        pass
    finally:
        receiver.stop()
        if ser:
            ser.close()
        cv2.destroyAllWindows()
        print("Inference loop stopped.")


if __name__ == '__main__':
    main()
