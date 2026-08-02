import cv2
import numpy as np
import pandas as pd
import os
import argparse
import sys
import torch
from torchvision import transforms
import cv2.aruco as aruco

# Add parent directory to path to import models
parent_dir = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from training.basic_cnn import BasicCNN

# Physical Platform Dimensions
PLATFORM_W = 187.5
PLATFORM_H = 142.0
CROP_PAD = 20

# ArUco Coordinates (Top-Left Origin)
MARKER_PHYSICAL_MM = {
    0: [12.0, 130.0],
    1: [175.5, 130.0],
    2: [175.5, 12.0],
    3: [12.0, 12.0],
    4: [12.0, 71.0],
    5: [175.5, 71.0],
}

PLATFORM_CORNERS_MM = np.array(
    [
        [[0.0, 0.0]],
        [[PLATFORM_W, 0.0]],
        [[PLATFORM_W, PLATFORM_H]],
        [[0.0, PLATFORM_H]],
    ],
    dtype=np.float32,
)


def main():
    parser = argparse.ArgumentParser(description="Extract CNN Sequential Features")
    parser.add_argument(
        "--session_dir",
        default="../../data/01_bronze/session_20260730_174916",
        help="Path to 01_bronze session",
    )
    parser.add_argument(
        "--out_dir",
        default="../../data/02_silver/session_20260730_174916",
        help="Directory to save the new dataset",
    )
    parser.add_argument(
        "--cnn_model",
        default="../models/cnn_2d_tracker_0730_v3/expert_tracker_best.pth",
        help="Path to best CNN weights",
    )
    args = parser.parse_args()

    session_dir = os.path.abspath(args.session_dir)
    out_dir = os.path.abspath(args.out_dir)
    cnn_path = os.path.abspath(args.cnn_model)

    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "cnn_sequential_features.csv")
    if os.path.exists(csv_path):
        os.remove(csv_path)
    # 1. Load Telemetry & Timestamps
    telemetry_path = os.path.join(session_dir, "telemetry.csv")
    timestamps_path = os.path.join(session_dir, "frame_timestamps.csv")
    video_path = os.path.join(session_dir, "rgb_video.mp4")

    if not (
        os.path.exists(telemetry_path)
        and os.path.exists(timestamps_path)
        and os.path.exists(video_path)
    ):
        print(f"Error: Missing required files in {session_dir}")
        return

    print(f"Merging telemetry from {session_dir}...")
    df_tel = pd.read_csv(telemetry_path).sort_values("host_timestamp_ms")
    df_ts = pd.read_csv(timestamps_path).sort_values("frame_timestamp_ms")

    merged = pd.merge_asof(
        df_ts,
        df_tel,
        left_on="frame_timestamp_ms",
        right_on="host_timestamp_ms",
        direction="nearest",
    )

    synced_telemetry = {}
    for _, row in merged.iterrows():
        f_idx = int(row["frame_index"])
        synced_telemetry[f_idx] = row

    # 2. Setup CNN
    print(f"Loading CNN from {cnn_path}...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cnn_model = BasicCNN().to(device)
    checkpoint = torch.load(cnn_path, map_location=device, weights_only=True)
    if "model_state_dict" in checkpoint:
        cnn_model.load_state_dict(checkpoint["model_state_dict"])
    else:
        cnn_model.load_state_dict(checkpoint)
    cnn_model.eval()

    preprocess = transforms.Compose(
        [
            transforms.Resize((240, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # 3. Setup ArUco
    try:
        dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        parameters = aruco.DetectorParameters()
        detector = aruco.ArucoDetector(dictionary, parameters)
    except AttributeError:
        dictionary = aruco.Dictionary_get(aruco.DICT_4X4_50)
        parameters = aruco.DetectorParameters_create()
        detector = None

    cap = cv2.VideoCapture(video_path)
    w_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_frame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    features = []
    frame_idx = 0
    saved_count = 0

    # State tracking to detect if ball fell off (frozen touch coordinates)
    prev_touch_x = None
    prev_touch_y = None

    print("Starting video processing loop...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in synced_telemetry:
            row = synced_telemetry[frame_idx]
            # Telemetry is already centered at 0,0
            touch_x = row["touch_x"]
            touch_y = row["touch_y"]

            # Filtering: If ball is off-board, telemetry freezes
            if prev_touch_x is not None:
                dist = (
                    (touch_x - prev_touch_x) ** 2 + (touch_y - prev_touch_y) ** 2
                ) ** 0.5
                if dist == 0.0 or dist > 30.0:
                    prev_touch_x = touch_x
                    prev_touch_y = touch_y
                    frame_idx += 1
                    continue

            prev_touch_x = touch_x
            prev_touch_y = touch_y

            # ArUco Detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if detector is not None:
                corners, ids, _ = detector.detectMarkers(gray)
            else:
                corners, ids, _ = aruco.detectMarkers(
                    gray, dictionary, parameters=parameters
                )

            if ids is not None and len(ids) >= 4:
                pixel_centers = []
                physical_centers = []
                ids_flat = np.ravel(ids)
                for i, m_id in enumerate(ids_flat):
                    if m_id in MARKER_PHYSICAL_MM:
                        center = np.mean(corners[i][0], axis=0)
                        pixel_centers.append(center)
                        physical_centers.append(MARKER_PHYSICAL_MM[m_id])

                if len(pixel_centers) >= 4:
                    pixel_centers = np.array(pixel_centers, dtype=np.float32)
                    physical_centers = np.array(physical_centers, dtype=np.float32)
                    M, _ = cv2.findHomography(pixel_centers, physical_centers)

                    if M is not None:
                        M_inv = np.linalg.inv(M)
                        pixel_corners = cv2.perspectiveTransform(
                            PLATFORM_CORNERS_MM, M_inv
                        )
                        xs = pixel_corners[:, 0, 0]
                        ys = pixel_corners[:, 0, 1]
                        x1 = int(max(0, xs.min() - CROP_PAD))
                        y1 = int(max(0, ys.min() - CROP_PAD))
                        x2 = int(min(w_frame, xs.max() + CROP_PAD))
                        y2 = int(min(h_frame, ys.max() + CROP_PAD))

                        if x2 > x1 and y2 > y1:
                            crop = frame[y1:y2, x1:x2]
                            # Force crop to RGB and expected tensor size
                            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                            from PIL import Image

                            pil_img = Image.fromarray(rgb_crop)
                            input_tensor = preprocess(pil_img).unsqueeze(0).to(device)

                            with torch.no_grad():
                                output = cnn_model(input_tensor)

                            norm_x, norm_y = output[0].cpu().numpy()
                            crop_w = x2 - x1
                            crop_h = y2 - y1

                            ball_crop_x = (norm_x + 1.0) * (crop_w / 2.0)
                            ball_crop_y = (norm_y + 1.0) * (crop_h / 2.0)

                            ball_frame_x = ball_crop_x + x1
                            ball_frame_y = ball_crop_y + y1

                            # Save CNN predictions directly as pixels
                            cnn_pixel_x = float(ball_frame_x)
                            cnn_pixel_y = float(ball_frame_y)

                            # Map ground truth physical MM back to pixels using M_inv
                            # Physical to Pixel mapping
                            # Convert center-relative telemetry to ArUco top-left relative
                            t_x = row["touch_x"]
                            t_y = row["touch_y"]

                            touch_x_topleft = t_x + (PLATFORM_W / 2.0)
                            touch_y_topleft = t_y + (PLATFORM_H / 2.0)

                            ball_pt_mm = np.array(
                                [[[touch_x_topleft, touch_y_topleft]]], dtype=np.float32
                            )
                            ball_pt_pixel = cv2.perspectiveTransform(ball_pt_mm, M_inv)[
                                0
                            ][0]

                            touch_pixel_x = float(ball_pt_pixel[0])
                            touch_pixel_y = float(ball_pt_pixel[1])

                            # Save all features mapping CNN pixel input to True State output
                            features.append(
                                {
                                    "frame_timestamp_ms": row["frame_timestamp_ms"],
                                    "cnn_pixel_x": cnn_pixel_x,
                                    "cnn_pixel_y": cnn_pixel_y,
                                    "target_x": row.get("target_x", 0.0),
                                    "target_y": row.get("target_y", 0.0),
                                    "touch_x": touch_x,
                                    "touch_y": touch_y,
                                    "touch_pixel_x": touch_pixel_x,
                                    "touch_pixel_y": touch_pixel_y,
                                    "deriv_x": row.get("deriv_x", 0.0),
                                    "deriv_y": row.get("deriv_y", 0.0),
                                }
                            )
                            saved_count += 1

                            if len(features) >= 100:
                                csv_path = os.path.join(
                                    out_dir, "cnn_sequential_features.csv"
                                )
                                out_df = pd.DataFrame(features)
                                header = not os.path.exists(csv_path)
                                out_df.to_csv(
                                    csv_path, mode="a", header=header, index=False
                                )
                                features = []
                                print(f"Extracted and saved {saved_count} frames...")

        frame_idx += 1

    cap.release()
    print(f"Extraction complete! Found {saved_count} valid frames.")

    if len(features) > 0:
        csv_path = os.path.join(out_dir, "cnn_sequential_features.csv")
        out_df = pd.DataFrame(features)
        header = not os.path.exists(csv_path)
        out_df.to_csv(csv_path, mode="a", header=header, index=False)
        print(f"Saved remaining frames to {csv_path}")


if __name__ == "__main__":
    main()
