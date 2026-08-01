import cv2
import pandas as pd
import argparse
import sys
import os
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description="Play through CNN sequential features to visually verify tracking on raw video."
    )
    parser.add_argument(
        "--data_dir",
        default="../../../data/02_silver/session_20260730_174916",
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--csv_name",
        default="cnn_sequential_features.csv",
        help="Name of the CNN features CSV file",
    )
    parser.add_argument(
        "--output", default="validation_labels.mp4", help="Output mp4 path"
    )
    parser.add_argument(
        "--max_frames", type=int, default=1500, help="Max frames to play"
    )
    parser.add_argument(
        "--bronze_dir",
        default="../../../data/01_bronze/session_20260730_174916",
        help="Path to original bronze dataset directory",
    )
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    bronze_dir = os.path.abspath(args.bronze_dir)
    csv_path = os.path.join(data_dir, args.csv_name)
    output_path = os.path.join(data_dir, args.output)

    orig_csv = os.path.join(bronze_dir, "frame_timestamps.csv")
    video_path = os.path.join(bronze_dir, "rgb_video.mp4")

    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}")
        sys.exit(1)
    if not os.path.exists(orig_csv):
        print(f"Error: Original frame_timestamps.csv not found at {orig_csv}")
        sys.exit(1)
    if not os.path.exists(video_path):
        print(f"Error: Original rgb_video.mp4 not found at {video_path}")
        sys.exit(1)

    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        print("Error: Dataset is empty.")
        sys.exit(1)

    print(f"Loading {orig_csv}...")
    orig_df = pd.read_csv(orig_csv)

    # Map frame_timestamp_ms -> feature row for O(1) lookup
    features_map = {}
    for _, row in df.iterrows():
        ts = row["frame_timestamp_ms"]
        features_map[ts] = row

    # Map frame_index -> frame_timestamp_ms
    idx_to_ts = {}
    for _, row in orig_df.iterrows():
        idx_to_ts[int(row["frame_index"])] = row["frame_timestamp_ms"]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        sys.exit(1)

    w_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_frame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (w_frame, h_frame))

    print(
        f"Generating playback for up to {args.max_frames if args.max_frames != -1 else 'all'} frames..."
    )

    frame_idx = 0
    saved_frames = 0

    pbar_total = (
        args.max_frames
        if args.max_frames != -1
        else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    )
    if pbar_total <= 0:
        pbar_total = None  # Fallback if frame count is unavailable
    pbar = tqdm(total=pbar_total)

    while cap.isOpened() and (args.max_frames == -1 or saved_frames < args.max_frames):
        ret, frame = cap.read()
        if not ret:
            break

        ts = idx_to_ts.get(frame_idx)
        if ts is not None and ts in features_map:
            row = features_map[ts]

            # Get Pixel Coordinates
            cnn_px = int(row.get("cnn_pixel_x", 0))
            cnn_py = int(row.get("cnn_pixel_y", 0))
            touch_px = int(row.get("touch_pixel_x", 0))
            touch_py = int(row.get("touch_pixel_y", 0))

            # Draw True Touch mapped by Homography (Red Circle)
            if 0 <= touch_px <= w_frame and 0 <= touch_py <= h_frame:
                cv2.circle(frame, (touch_px, touch_py), 15, (0, 0, 255), 3)

            # Draw CNN Prediction (Blue Circle)
            if 0 <= cnn_px <= w_frame and 0 <= cnn_py <= h_frame:
                cv2.circle(frame, (cnn_px, cnn_py), 15, (255, 0, 0), 3)

            # Annotations
            cv2.putText(
                frame,
                "Red: True Telemetry | Blue: CNN Pred",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"CNN Pixels X:{cnn_px} Y:{cnn_py}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Touch Pixels X:{touch_px} Y:{touch_py}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Video Frame: {frame_idx}",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            out.write(frame)
            saved_frames += 1
            pbar.update(1)

        frame_idx += 1

    pbar.close()
    cap.release()
    out.release()
    print(f"Done! Saved verification video to {output_path}")


if __name__ == "__main__":
    main()
