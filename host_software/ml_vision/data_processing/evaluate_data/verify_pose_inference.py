import cv2
import pandas as pd
import os
import argparse
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--session_dir",
        required=True,
        help="Path to session dir (e.g. host_software/data/02_silver/session_20260728_102908)",
    )
    parser.add_argument("--out_video", default="validation_pose.mp4")
    args = parser.parse_args()

    csv_path = os.path.join(args.session_dir, "yolo_platform_corners_features.csv")
    if not os.path.exists(csv_path):
        print(
            f"Error: {csv_path} not found. Run auto_label_telemetry_with_pose.py first."
        )
        return

    df = pd.read_csv(csv_path)

    out_video_path = os.path.join(args.session_dir, args.out_video)
    print(f"Generating validation video: {out_video_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = None

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Writing frames"):
        img_name = row["image_file"]
        img_path = os.path.join(args.session_dir, "images", img_name)

        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        if out is None:
            h, w = img.shape[:2]
            out = cv2.VideoWriter(out_video_path, fourcc, 30.0, (w, h))

        # Draw the projected ball bounding box
        bx1 = int(row["ball_xmin"])
        by1 = int(row["ball_ymin"])
        bx2 = int(row["ball_xmax"])
        by2 = int(row["ball_ymax"])

        # Red box for ball
        cv2.rectangle(img, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
        cv2.putText(
            img,
            "Projected Ball",
            (bx1, by1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )

        # We don't have the explicit 4 corner pixels saved in the CSV!
        # But we DO have the cropped image which we can also visualize if we want.
        # Let's just write the uncropped frame with the projected ball. If the ball tracks the physical ball perfectly, the homography (and thus the corners) was perfect!

        out.write(img)

    if out is not None:
        out.release()
        print("Done!")
    else:
        print("No frames were written.")


if __name__ == "__main__":
    main()
