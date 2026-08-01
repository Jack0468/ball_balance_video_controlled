import pandas as pd


def merge_datasets(csv_path_1, csv_path_2, output_path):
    print(f"Loading {csv_path_1}...")
    try:
        df1 = pd.read_csv(csv_path_1)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_path_1}")
        return

    print(f"Loading {csv_path_2}...")
    try:
        df2 = pd.read_csv(csv_path_2)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_path_2}")
        return

    print("Merging datasets...")
    # pd.concat automatically aligns columns.
    # ignore_index=True resets the row numbers so they flow sequentially 0 to N.
    merged_df = pd.concat([df1, df2], ignore_index=True)

    # Save to the new CSV
    merged_df.to_csv(output_path, index=False)
    print(f"Success! Merged dataset saved to {output_path}")
    print(f"Total rows in new dataset: {len(merged_df)}")


# ==========================================
# Configuration and Execution
# ==========================================
if __name__ == "__main__":
    # Define your file paths here
    DATASET_1_PATH = (
        "./images_iphone/labels_normalized.csv"  # The CSV WITH frame_timestamp_ms
    )
    DATASET_2_PATH = "./session_20260728_102908./labels_normalized.csv"  # The CSV WITHOUT frame_timestamp_ms
    OUTPUT_PATH = (
        "./cropped_yolo/labels_normalized.csv"  # The name of your new combined file
    )

    merge_datasets(DATASET_1_PATH, DATASET_2_PATH, OUTPUT_PATH)
