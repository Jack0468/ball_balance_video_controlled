import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import numpy as np
import pandas as pd
import cv2
import argparse
import sys
from ultralytics import YOLO

parent_dir = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from core.coordinate_math import HomographyProjector


def extract_features():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir_input",
        default=None,
        help="Path to a session folder containing images/ and labels.csv",
    )
    parser.add_argument(
        "--data_dir", default="../../data/02_silver", help="Path to telemetry dataset"
    )
    parser.add_argument(
        "--model_path",
        default="../models/yolov8_platform_pose_markers_0728_v4/weights/best.pt",
        help="Path to YOLO model",
    )
    parser.add_argument(
        "--num_train",
        type=int,
        default=5000,
        help="Number of training samples to extract",
    )
    parser.add_argument(
        "--num_test", type=int, default=1000, help="Number of test samples to extract"
    )
    parser.add_argument(
        "--csv_name",
        default="labels.csv",
        help="Name of the telemetry csv file in dir_input",
    )
    parser.add_argument(
        "--out_csv", default="yolo_features.csv", help="Name of the output csv file"
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.abspath(os.path.join(script_dir, args.model_path))

    if args.dir_input:
        data_dir = os.path.abspath(args.dir_input)
        csv_path = os.path.join(data_dir, args.csv_name)
        images_dir = os.path.join(data_dir, "images")
    else:
        if os.path.exists(args.data_dir):
            data_dir = os.path.abspath(args.data_dir)
        else:
            data_dir = os.path.abspath(os.path.join(script_dir, args.data_dir))
        csv_path = os.path.join(data_dir, "labels_normalized.csv")
        images_dir = os.path.join(data_dir, "images")

    print(f"Loading YOLO Model from {model_path}...")
    model = YOLO(model_path)

    print(f"Loading Telemetry from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Split into train/test (first 80% train, last 20% test to match evaluation script)
    split_idx = int(0.8 * len(df))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    # Randomly sample to save time
    if args.num_train != -1 and len(train_df) > args.num_train:
        train_df = train_df.sample(n=args.num_train, random_state=42)
    if args.num_test != -1 and len(test_df) > args.num_test:
        test_df = test_df.sample(n=args.num_test, random_state=42)

    print(
        f"Extracting features for {len(train_df)} train frames and {len(test_df)} test frames."
    )

    dst_pts = np.array([[-70, 55], [70, 55], [70, -55], [-70, -55]], dtype=np.float32)
    projector = HomographyProjector(dst_pts)

    out_csv = os.path.join(data_dir, args.out_csv)
    processed_images = set()
    if os.path.exists(out_csv):
        existing_df = pd.read_csv(out_csv)
        if "image_file" in existing_df.columns:
            processed_images = set(existing_df["image_file"].values)
            print(
                f"Found {len(processed_images)} already processed frames in {out_csv}. Resuming..."
            )

    def process_subset(subset_df, split_name):
        features = []
        count = 0
        total = len(subset_df)
        for idx, row in subset_df.iterrows():
            count += 1

            if row["image_file"] in processed_images:
                if count % 500 == 0:
                    print(
                        f"Skipped {count}/{total} {split_name} frames (already processed)..."
                    )
                continue

            img_path = os.path.join(images_dir, row["image_file"])
            if not os.path.exists(img_path):
                continue

            img = cv2.imread(img_path)
            if img is None:
                continue

            results = model.predict(source=img, imgsz=640, conf=0.5, verbose=False)
            if not results or len(results) == 0:
                continue

            res = results[0]
            if res.boxes is None:
                continue

            classes = res.boxes.cls.cpu().numpy()
            boxes = res.boxes.xywh.cpu().numpy()

            corners = None
            ball_box = None
            markers = {i: None for i in range(2, 13)}

            for i, cls in enumerate(classes):
                c = int(cls)
                if c == 0:  # Platform
                    if res.keypoints is not None and len(res.keypoints.xy) > i:
                        kpts = res.keypoints.xy[i].cpu().numpy()
                        if len(kpts) == 4:
                            corners = kpts
                elif c == 1:  # Ball
                    ball_box = boxes[i]
                elif c >= 2 and c <= 12:  # Markers
                    markers[c] = boxes[i]

            if corners is not None:
                homography_x, homography_y = 0.0, 0.0
                ball_x, ball_y, ball_w, ball_h = -1.0, -1.0, -1.0, -1.0
                ball_present = 0.0

                if ball_box is not None:
                    ball_x, ball_y, ball_w, ball_h = (
                        ball_box[0],
                        ball_box[1],
                        ball_box[2],
                        ball_box[3],
                    )
                    ball_present = 1.0

                    if projector.update_homography(corners):
                        hx, hy = projector.project_point(ball_x, ball_y)
                        if hx is not None and hy is not None:
                            homography_x, homography_y = hx, hy

                feature_dict = {
                    "image_file": row["image_file"],
                    "split": split_name,
                    "ball_present": ball_present,
                    "ball_x": ball_x,
                    "ball_y": ball_y,
                    "ball_w": ball_w,
                    "ball_h": ball_h,
                    "kpt0_x": corners[0][0],
                    "kpt0_y": corners[0][1],
                    "kpt1_x": corners[1][0],
                    "kpt1_y": corners[1][1],
                    "kpt2_x": corners[2][0],
                    "kpt2_y": corners[2][1],
                    "kpt3_x": corners[3][0],
                    "kpt3_y": corners[3][1],
                    "homography_x": homography_x,
                    "homography_y": homography_y,
                    "touch_x": row["touch_x"],
                    "touch_y": row["touch_y"],
                }

                for c in range(2, 13):
                    if markers[c] is not None:
                        feature_dict[f"marker{c}_x"] = markers[c][0]
                        feature_dict[f"marker{c}_y"] = markers[c][1]
                    else:
                        feature_dict[f"marker{c}_x"] = -1.0
                        feature_dict[f"marker{c}_y"] = -1.0

                features.append(feature_dict)

            if len(features) >= 500:
                out_df = pd.DataFrame(features)
                header = not os.path.exists(out_csv)
                out_df.to_csv(out_csv, mode="a", header=header, index=False)
                features = []
                print(f"Processed {count}/{total} {split_name} frames...")

        if len(features) > 0:
            out_df = pd.DataFrame(features)
            header = not os.path.exists(out_csv)
            out_df.to_csv(out_csv, mode="a", header=header, index=False)

    process_subset(train_df, "train")
    process_subset(test_df, "test")

    print(f"Finished extracting all feature vectors to {out_csv}")


if __name__ == "__main__":
    extract_features()
