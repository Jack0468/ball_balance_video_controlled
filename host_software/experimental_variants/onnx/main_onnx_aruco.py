"""EXPERIMENTAL ENTRY POINT: ONNX-accelerated Vision + ArUco. Runs YOLO via ONNX runtime for faster CPU/Edge inference without PyTorch overhead."""

import cv2
import time
import numpy as np
import os
import sys
import serial
import argparse
import collections
import cv2.aruco as aruco
import onnxruntime as ort

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from src.receivers import USBReceiver, UDPReceiver
from src.utils import find_stm32_port

# --- Configuration ---
SERIAL_PORT = "COM7"
SERIAL_BAUD = 2000000
CROP_PAD = 20  # Pixels of padding around the platform crop

# True physical plate bounds
PLATFORM_W = 187.5
PLATFORM_H = 142.0

# These are the exact millimeter coordinates of the centers of the 6 markers
# relative to the Top-Left (0,0) corner of the printed PDF bounding box.
MARKER_PHYSICAL_MM = {
    0: [12.0, 130.0],
    1: [175.5, 130.0],
    2: [175.5, 12.0],
    3: [12.0, 12.0],
    4: [12.0, 71.0],
    5: [175.5, 71.0],
}

# The 4 physical corners of the platform boundary
PLATFORM_CORNERS_MM = np.array(
    [
        [[0.0, 0.0]],
        [[PLATFORM_W, 0.0]],
        [[PLATFORM_W, PLATFORM_H]],
        [[0.0, PLATFORM_H]],
    ],
    dtype=np.float32,
)
# ---------------------


def preprocess_numpy(img):
    # cv2 uses (width, height) for resize
    img = cv2.resize(img, (320, 240))
    img = img.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std

    # HWC to CHW
    img = np.transpose(img, (2, 0, 1))

    # Add batch dim -> (1, 3, 240, 320)
    return np.expand_dims(img, axis=0)


def main():
    parser = argparse.ArgumentParser(
        description="Cascaded ArUco -> CNN -> MLP Ball Tracker (ONNX)"
    )
    parser.add_argument("--cam_id", type=int, default=1, help="Camera ID for USB mode")
    parser.add_argument(
        "--port", type=str, default="auto", help="STM32 serial port or 'auto'"
    )
    parser.add_argument("--udp", action="store_true", help="Use UDP receiver")
    parser.add_argument("--udp_port", type=int, default=5001, help="UDP listen port")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Disable GUI display (improves performance)",
    )
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

    # ---- 1. Model Init (ONNX) ----
    script_dir = os.path.dirname(os.path.abspath(__file__))

    cnn_path = os.path.abspath(
        os.path.join(
            script_dir, "ml_vision/models/cnn_2d_tracker_0730_v3/expert_tracker_best.onnx"
        )
    )
    mlp_path = os.path.abspath(
        os.path.join(
            script_dir,
            "ml_vision/models/mlp_corrector_time_varuco_0730_v1/mlp_corrector_best.onnx",
        )
    )

    if not os.path.exists(cnn_path):
        print(f"Error: ONNX CNN model not found at {cnn_path}")
        print("Please run export_to_onnx.py first!")
        return

    if not os.path.exists(mlp_path):
        print(f"Error: ONNX MLP model not found at {mlp_path}")
        print("Please run export_to_onnx.py first!")
        return

    # Initialize ONNX Sessions
    print("Loading ONNX sessions...")
    cnn_session = ort.InferenceSession(cnn_path, providers=["CPUExecutionProvider"])
    mlp_session = ort.InferenceSession(mlp_path, providers=["CPUExecutionProvider"])

    cnn_input_name = cnn_session.get_inputs()[0].name
    mlp_input_name = mlp_session.get_inputs()[0].name

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

    print(f"Starting ArUco -> CNN -> MLP Tracker loop... (press Ctrl+C to quit)")

    # Time-Series History Buffer for the MLP
    history_buffer = collections.deque(maxlen=1)
    last_frame_time = time.perf_counter()

    try:
        while True:
            frame = receiver.get_latest_frame()
            if frame is None:
                continue

            start_t = time.perf_counter()
            dt_ms = (start_t - last_frame_time) * 1000.0
            last_frame_time = start_t

            # Bound dt for safety if there's a huge lag spike
            if dt_ms > 100.0:
                dt_ms = 33.0
                history_buffer.clear()  # Clear buffer on large skips

            h_frame, w_frame = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # --- STAGE 1: ArUco Homography ---
            aruco_t0 = time.perf_counter()
            if detector is not None:
                corners, ids, rejected = detector.detectMarkers(gray)
            else:
                corners, ids, rejected = aruco.detectMarkers(
                    gray, dictionary, parameters=parameters
                )

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
            x1 = int(max(0, xs.min() - CROP_PAD))
            y1 = int(max(0, ys.min() - CROP_PAD))
            x2 = int(min(w_frame, xs.max() + CROP_PAD))
            y2 = int(min(h_frame, ys.max() + CROP_PAD))

            if x2 <= x1 or y2 <= y1:
                print("Degenerate crop — skipping")
                continue

            crop = frame[y1:y2, x1:x2]

            # --- STAGE 3: CNN Ball Tracker (ONNX) ---
            cnn_t0 = time.perf_counter()
            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            input_tensor = preprocess_numpy(rgb_crop)

            # ONNX Inference
            output = cnn_session.run(None, {cnn_input_name: input_tensor})[0]

            cnn_ms = (time.perf_counter() - cnn_t0) * 1000.0

            # The CNN predicts [-1, 1] relative to the crop dimensions!
            norm_x, norm_y = output[0]

            crop_w = x2 - x1
            crop_h = y2 - y1

            # Convert [-1, 1] back to [0, crop_w] and [0, crop_h]
            ball_crop_x = (norm_x + 1.0) * (crop_w / 2.0)
            ball_crop_y = (norm_y + 1.0) * (crop_h / 2.0)

            # Convert crop pixels to full frame pixels
            ball_frame_x = ball_crop_x + x1
            ball_frame_y = ball_crop_y + y1

            # Use Homography (M) to map frame pixels to physical platform mm!
            ball_pt = np.array([[[ball_frame_x, ball_frame_y]]], dtype=np.float32)
            touch_pt = cv2.perspectiveTransform(ball_pt, M)

            # touch_x and touch_y are now perfectly in platform millimeters (0 to PLATFORM_W)
            touch_x_raw = float(touch_pt[0, 0, 0])
            touch_y_raw = float(touch_pt[0, 0, 1])

            # We must center the target relative to the PID firmware!
            centered_touch_x = touch_x_raw - (PLATFORM_W / 2.0)
            centered_touch_y = touch_y_raw - (PLATFORM_H / 2.0)

            # --- STAGE 4: MLP Time Corrector (ONNX) ---
            mlp_t0 = time.perf_counter()

            # Normalize inputs matching the V2 MLP training!
            norm_cnn_x = (ball_frame_x / 320.0) - 1.0
            norm_cnn_y = (ball_frame_y / 240.0) - 1.0
            norm_target_x = 0.0  # Target is always 0.0 in PID mode
            norm_target_y = 0.0
            norm_dt = (dt_ms / 33.0) - 1.0

            history_buffer.append(
                [norm_cnn_x, norm_cnn_y, norm_target_x, norm_target_y, norm_dt]
            )

            mlp_ms = 0.0
            if len(history_buffer) == 1:
                # Flatten the 1x5 buffer into a 1x5 input array
                mlp_input = (
                    np.array(history_buffer, dtype=np.float32).flatten().reshape(1, -1)
                )
                mlp_out = mlp_session.run(None, {mlp_input_name: mlp_input})[0][0]

                final_x = float(mlp_out[0])
                final_y = float(mlp_out[1])
                deriv_x, deriv_y = 0.0, 0.0

                mlp_ms = (time.perf_counter() - mlp_t0) * 1000.0
            else:
                # Buffer not full, fallback to raw CNN tracking
                final_x = centered_touch_x
                final_y = centered_touch_y
                deriv_x, deriv_y = 0.0, 0.0

            # ----------------------------------------------------------------
            # Serial Transmission
            # ----------------------------------------------------------------
            try:
                # Current compatible payload:
                payload = f"{final_x:.2f},{final_y:.2f},0.00,0.00\n".encode("ascii")

                # TODO: When firmware is updated to support derivatives directly, swap to this:
                # payload = f"{final_x:.2f},{final_y:.2f},{deriv_x:.2f},{deriv_y:.2f}\n".encode('ascii')

                if ser is not None:
                    ser.write(payload)
            except Exception as e:
                print(f"Serial Error: {e}")

            end_t = time.perf_counter()
            total_ms = (end_t - start_t) * 1000.0
            fps = 1.0 / (end_t - start_t)

            print(
                f"Ball: X={final_x:+6.1f} Y={final_y:+6.1f} mm | Target: 0.0,0.0 | "
                f"FPS: {fps:.1f} | Total={total_ms:.1f}ms (ArUco={aruco_ms:.1f}ms, CNN={cnn_ms:.1f}ms, MLP={mlp_ms:.1f}ms)"
            )

            # Optional: Display for debugging
            if not args.headless:
                cv2.circle(
                    frame, (int(ball_frame_x), int(ball_frame_y)), 10, (0, 0, 255), -1
                )
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.imshow("ArUco + CNN + MLP Tracker (ONNX)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        pass
    finally:
        receiver.stop()
        if ser:
            ser.close()
        cv2.destroyAllWindows()
        print("Inference loop stopped.")


if __name__ == "__main__":
    main()
