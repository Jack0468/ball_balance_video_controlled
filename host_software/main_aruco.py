import cv2
import time
import numpy as np
import os
import sys
import serial
import argparse
import torch
import cv2.aruco as aruco
from torchvision import transforms

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from src.receivers import USBReceiver, UDPReceiver
from src.utils import find_stm32_port
from ml_vision.training.basic_cnn import BasicCNN

# --- Configuration ---
SERIAL_PORT       = "COM7"
SERIAL_BAUD       = 2000000
CROP_PAD          = 20      # Pixels of padding around the platform crop

# True physical plate bounds
PLATFORM_W = 187.5
PLATFORM_H = 142.0

# These are the exact millimeter coordinates of the centers of the 6 markers
# relative to the Top-Left (0,0) corner of the printed PDF bounding box.
MARKER_PHYSICAL_MM = {
    0: [12.0, 12.0],
    1: [175.5, 12.0],
    2: [175.5, 130.0],
    3: [12.0, 130.0],
    4: [12.0, 71.0],
    5: [175.5, 71.0]
}

# The 4 physical corners of the platform boundary
PLATFORM_CORNERS_MM = np.array([
    [[0.0, 0.0]],
    [[PLATFORM_W, 0.0]],
    [[PLATFORM_W, PLATFORM_H]],
    [[0.0, PLATFORM_H]]
], dtype=np.float32)
# ---------------------

def load_cnn_tracker(model_path, device):
    model = BasicCNN(num_outputs=2)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser(description="Cascaded ArUco -> CNN Ball Tracker")
    parser.add_argument("--cam_id",   type=int, default=1,    help="Camera ID for USB mode")
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
    script_dir = os.path.dirname(os.path.abspath(__file__))

    cnn_path = os.path.abspath(os.path.join(script_dir, 'ml_vision/models/cnn_2d_tracker_v2/expert_tracker_best.pth'))
    cnn_model = load_cnn_tracker(cnn_path, device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((240, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # ---- 2. ArUco Init ----
    try:
        dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        parameters = aruco.DetectorParameters()
        detector = aruco.ArucoDetector(dictionary, parameters)
    except AttributeError:
        dictionary = aruco.Dictionary_get(aruco.DICT_4X4_50)
        parameters = aruco.DetectorParameters_create()
        detector = None

    # ---- 3. Serial Port Init ----
    try:
        ser = serial.Serial(args.port, SERIAL_BAUD, timeout=0)
        print(f"Connected to STM32 on {args.port} at {SERIAL_BAUD} baud.")
    except Exception:
        print(f"Could not open serial port {args.port}. Continuing in dry-run mode.")
        ser = None

    # ---- 4. Camera/Receiver Init ----
    if args.udp:
        receiver = UDPReceiver(port=args.udp_port, width=640, height=480)
    else:
        receiver = USBReceiver(camera_id=args.cam_id)

    print("Waiting for camera feed...")
    frame = None
    while frame is None:
        frame = receiver.get_latest_frame()
        time.sleep(0.1)

    print(f"Starting ArUco -> CNN Tracker loop... (press Ctrl+C to quit)")

    try:
        while True:
            frame = receiver.get_latest_frame()
            if frame is None:
                continue

            start_t = time.perf_counter()
            h_frame, w_frame = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # --- STAGE 1: ArUco Homography ---
            aruco_t0 = time.perf_counter()
            if detector is not None:
                corners, ids, rejected = detector.detectMarkers(gray)
            else:
                corners, ids, rejected = aruco.detectMarkers(gray, dictionary, parameters=parameters)
            
            M = None
            if ids is not None:
                ids = ids.flatten()
                pixel_centers = []
                physical_centers = []
                
                for i, marker_id in enumerate(ids):
                    if marker_id in MARKER_PHYSICAL_MM:
                        marker_corners = corners[i][0]
                        center = np.mean(marker_corners, axis=0)
                        pixel_centers.append(center)
                        physical_centers.append(MARKER_PHYSICAL_MM[marker_id])
                
                if len(pixel_centers) >= 4:
                    pixel_centers = np.array(pixel_centers, dtype=np.float32)
                    physical_centers = np.array(physical_centers, dtype=np.float32)
                    M, _ = cv2.findHomography(pixel_centers, physical_centers)
            
            aruco_ms = (time.perf_counter() - aruco_t0) * 1000.0

            if M is None:
                print(f"Gate closed - Insufficient ArUco markers ({aruco_ms:.1f}ms)")
                continue

            # --- STAGE 2: Crop to Platform ---
            M_inv = np.linalg.inv(M)
            # Project the 4 physical corners of the board back to pixels
            pixel_corners = cv2.perspectiveTransform(PLATFORM_CORNERS_MM, M_inv)
            
            xs = pixel_corners[:, 0, 0]
            ys = pixel_corners[:, 0, 1]
            x1 = int(max(0,        xs.min() - CROP_PAD))
            y1 = int(max(0,        ys.min() - CROP_PAD))
            x2 = int(min(w_frame,  xs.max() + CROP_PAD))
            y2 = int(min(h_frame,  ys.max() + CROP_PAD))
            
            if x2 <= x1 or y2 <= y1:
                print("Degenerate crop — skipping")
                continue
                
            crop = frame[y1:y2, x1:x2]
            
            # --- STAGE 3: CNN Ball Tracker ---
            cnn_t0 = time.perf_counter()
            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            input_tensor = preprocess(rgb_crop).unsqueeze(0).to(device)

            with torch.no_grad():
                output = cnn_model(input_tensor)

            cnn_ms = (time.perf_counter() - cnn_t0) * 1000.0

            # The CNN now predicts the physical touch pad telemetry directly in [-1, 1] space!
            norm_x, norm_y = output[0].cpu().numpy()
            
            # Convert [-1, 1] directly to physical mm (0 to PLATFORM_W)
            touch_x = (norm_x + 1.0) * (PLATFORM_W / 2.0)
            touch_y = (norm_y + 1.0) * (PLATFORM_H / 2.0)
            
            # For visual debugging, we use the Inverse Homography (M_inv) 
            # to map the physical mm back to pixels on the webcam frame!
            touch_pt = np.array([[[touch_x, touch_y]]], dtype=np.float32)
            frame_pt = cv2.perspectiveTransform(touch_pt, M_inv)
            ball_frame_x = float(frame_pt[0, 0, 0])
            ball_frame_y = float(frame_pt[0, 0, 1])

            # ----------------------------------------------------------------
            # Serial Transmission
            # We must center the target relative to the PID firmware!
            # The firmware expects the center to be (0,0). So we shift origin:
            # ----------------------------------------------------------------
            centered_touch_x = touch_x - (PLATFORM_W / 2.0)
            centered_touch_y = touch_y - (PLATFORM_H / 2.0)
            
            try:
                # payload: cam_x, cam_y, target_x, target_y
                payload = f"{centered_touch_x:.2f},{centered_touch_y:.2f},0.00,0.00\n".encode('ascii')
                if ser is not None:
                    ser.write(payload)
            except Exception as e:
                print(f"Serial Error: {e}")

            end_t     = time.perf_counter()
            total_ms  = (end_t - start_t) * 1000.0
            fps       = 1.0 / (end_t - start_t)

            print(
                f"Ball: X={centered_touch_x:+6.1f} Y={centered_touch_y:+6.1f} mm | Target: centre (0,0) | "
                f"FPS: {fps:.1f} | Total={total_ms:.1f}ms (ArUco={aruco_ms:.1f}ms, CNN={cnn_ms:.1f}ms)"
            )
            
            # Optional: Display for debugging
            cv2.circle(frame, (int(ball_frame_x), int(ball_frame_y)), 10, (0, 0, 255), -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.imshow("ArUco + CNN Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

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
