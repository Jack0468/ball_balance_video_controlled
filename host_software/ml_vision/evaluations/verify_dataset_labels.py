import os
import cv2
import pandas as pd
import argparse
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description="Verify Dataset Labels via Video Generation"
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Path to data directory (e.g. host_software/data/02_silver/session_20260730_174916)",
    )
    parser.add_argument(
        "--csv_name", default="yolo_features.csv", help="Name of the CSV labels file"
    )
    parser.add_argument(
        "--output_name",
        default="validation_labels.mp4",
        help="Name of the output video file",
    )
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    csv_path = os.path.join(data_dir, args.csv_name)
    images_dir = os.path.join(data_dir, "images")
    output_path = os.path.join(data_dir, args.output_name)

    if not os.path.exists(csv_path):
        print(f"ERROR: CSV file not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # Filter for frames where the ball is present
    if "ball_present" in df.columns:
        df = df[df["ball_present"] == 1.0].reset_index(drop=True)

    if df.empty:
        print("No valid labeled frames found in the dataset.")
        return

    # Read the first image to get dimensions
    first_img_path = os.path.join(images_dir, df.iloc[0]["image_file"])
    first_img = cv2.imread(first_img_path)
    if first_img is None:
        print(f"ERROR: Could not read first image at {first_img_path}")
        return

    height, width, _ = first_img.shape

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

    print(f"Rendering validation video to {output_path}...")

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        img_name = row["image_file"]
        img_path = os.path.join(images_dir, img_name)

        frame = cv2.imread(img_path)
        if frame is None:
            continue

        # Ensure we resize to match video dimensions if crops vary slightly
        frame = cv2.resize(frame, (width, height))

        # Read the inverse-homography calculated pixel labels
        ball_x = float(row["ball_x"])
        ball_y = float(row["ball_y"])

        if not pd.isna(ball_x) and not pd.isna(ball_y):
            # Draw a highly visible red dot at the label coordinate
            cv2.circle(frame, (int(ball_x), int(ball_y)), 6, (0, 0, 255), -1)
            cv2.circle(
                frame, (int(ball_x), int(ball_y)), 8, (255, 255, 255), 2
            )  # White outline

            # Put text for coordinate reference
            cv2.putText(
                frame,
                f"Label: ({int(ball_x)}, {int(ball_y)})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        out_video.write(frame)

    out_video.release()
    print(f"Done! Open {output_path} to verify the inverse homography labels visually.")


if __name__ == "__main__":
    main()
