import os
import pandas as pd
from PIL import Image


def generate_ground_truth_crops():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_base = os.path.abspath(os.path.join(script_dir, "..", "..", "data"))

    datasets = [
        {
            "features_csv": os.path.join(
                data_base, "02_silver", "session_20260728_102908", "yolo_features.csv"
            ),
            "input_images": os.path.join(
                data_base, "02_silver", "session_20260728_102908", "images"
            ),
            "output_dir": os.path.join(
                data_base, "02_silver", "session_20260728_102908", "cropped"
            ),
        },
        {
            "features_csv": os.path.join(
                data_base, "03_gold", "images_iphone", "yolo_features.csv"
            ),
            "input_images": os.path.join(
                data_base, "02_silver", "images_iphone", "images"
            ),
            "output_dir": os.path.join(data_base, "03_gold_cropped", "images_iphone"),
        },
    ]

    for ds in datasets:
        features_csv = ds["features_csv"]
        input_images = ds["input_images"]
        output_dir = ds["output_dir"]

        if not os.path.exists(features_csv):
            print(f"Skipping {features_csv} (not found)")
            continue

        print(f"\nProcessing {features_csv} ...")
        df = pd.read_csv(features_csv)

        out_images_dir = os.path.join(output_dir, "images")
        os.makedirs(out_images_dir, exist_ok=True)

        labels_normalized = []

        count = 0
        for _, row in df.iterrows():
            image_file = row["image_file"]
            img_path = os.path.join(input_images, image_file)

            if not os.path.exists(img_path):
                print(f"Warning: Image not found - {img_path}")
                continue

            try:
                kpts_x = [row["kpt0_x"], row["kpt1_x"], row["kpt2_x"], row["kpt3_x"]]
                kpts_y = [row["kpt0_y"], row["kpt1_y"], row["kpt2_y"], row["kpt3_y"]]

                with Image.open(img_path) as img:
                    img_width, img_height = img.size

                    left = int(min(kpts_x))
                    right = int(max(kpts_x))
                    top = int(min(kpts_y))
                    bottom = int(max(kpts_y))

                    left = max(0, left)
                    top = max(0, top)
                    right = min(img_width, right)
                    bottom = min(img_height, bottom)

                    if right <= left or bottom <= top:
                        print(f"Warning: Invalid crop box for {image_file}")
                        continue

                    cropped_img = img.crop((left, top, right, bottom))
                    cropped_img.save(os.path.join(out_images_dir, image_file))

                    crop_w = right - left
                    crop_h = bottom - top

                    # Calculate normalized target
                    target_x = (row["ball_x"] - left) / crop_w
                    target_y = (row["ball_y"] - top) / crop_h

                    # Ensure targets are strictly bounded 0 to 1
                    target_x = max(0.0, min(1.0, target_x))
                    target_y = max(0.0, min(1.0, target_y))

                    # Build row for labels_normalized.csv (mirroring silver format for compatibility)
                    labels_normalized.append(
                        {
                            "image_file": image_file,
                            "split": row["split"],
                            "target_x": target_x,
                            "target_y": target_y,
                            "touch_x": row["touch_x"],
                            "touch_y": row["touch_y"],
                        }
                    )
                    count += 1
            except Exception as e:
                print(f"Error processing {image_file}: {e}")

        if labels_normalized:
            out_csv = os.path.join(output_dir, "labels_normalized.csv")
            pd.DataFrame(labels_normalized).to_csv(out_csv, index=False)
            print(f"Saved {count} crops and labels to {output_dir}")


if __name__ == "__main__":
    generate_ground_truth_crops()
