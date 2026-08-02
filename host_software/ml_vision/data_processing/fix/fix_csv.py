import os
import pandas as pd

# 1. Define your paths
csv_path = "cropped_yolo/labels_normalized.csv"
images_dir = "cropped_yolo/images"
cleaned_csv_path = "cropped_yolo/labels_cleaned.csv"

print("Loading original CSV...")
df = pd.read_csv(csv_path)
valid_rows = []
missing_count = 0

print(f"Checking {len(df)} rows to ensure images exist...")

# 2. Check each image
for index, row in df.iterrows():
    img_name = row["image_file"]

    # NOTE: If your images are in train/test folders, you might need to do:
    # img_path = os.path.join(images_dir, row['split'], img_name)
    img_path = os.path.join(images_dir, img_name)

    if os.path.exists(img_path):
        valid_rows.append(row)
    else:
        missing_count += 1

# 3. Save the cleaned data
cleaned_df = pd.DataFrame(valid_rows)
cleaned_df.to_csv(cleaned_csv_path, index=False)

print(f"Done! Found {missing_count} missing images.")
print(f"Saved cleaned CSV with {len(cleaned_df)} valid rows to {cleaned_csv_path}")
