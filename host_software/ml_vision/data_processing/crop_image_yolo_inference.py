import os
import pandas as pd
from PIL import Image

def crop_platform_from_csv(csv_path, input_base_dir, output_base_dir):
    # 1. Read the CSV file
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find CSV file at {csv_path}")
        return

    # 2. Iterate through each row in the dataset
    for index, row in df.iterrows():
        image_file = row['image_file']
        split = row['split']
        
        # Extract the 4 keypoints (X and Y coordinates)
        try:
            kpts_x = [
                float(row['kpt0_x']), float(row['kpt1_x']), 
                float(row['kpt2_x']), float(row['kpt3_x'])
            ]
            kpts_y = [
                float(row['kpt0_y']), float(row['kpt1_y']), 
                float(row['kpt2_y']), float(row['kpt3_y'])
            ]
        except ValueError as e:
            print(f"Warning: Missing or invalid keypoint data for {image_file}. Skipping.")
            continue
        
        # 3. Construct paths based on the 'split' column
        img_path = os.path.join(input_base_dir, image_file)
        
        if not os.path.exists(img_path):
            print(f"Warning: Image not found - {img_path}. Skipping.")
            continue
            
        try:
            # Open the image
            with Image.open(img_path) as img:
                img_width, img_height = img.size
                
                # 4. Calculate the straight rectangular bounding box encompassing all 4 keypoints
                left = int(min(kpts_x))
                right = int(max(kpts_x))
                top = int(min(kpts_y))
                bottom = int(max(kpts_y))
                
                # 5. Enforce image boundaries to prevent crashing if points go slightly off-screen
                left = max(0, left)
                top = max(0, top)
                right = min(img_width, right)
                bottom = min(img_height, bottom)
                
                # Validate the crop box (ensure it has area)
                if right <= left or bottom <= top:
                    print(f"Warning: Invalid crop box for {image_file}. Skipping.")
                    continue
                
                # 6. Crop the image
                cropped_img = img.crop((left, top, right, bottom))
                
                # Create output directory for this split if it doesn't exist
                output_split_dir = os.path.join(output_base_dir, split)
                os.makedirs(output_split_dir, exist_ok=True)
                
                # Save the cropped image
                output_path = os.path.join(output_split_dir, image_file)
                cropped_img.save(output_path)
                #print(f"Successfully cropped platform: {output_path}")
                
        except Exception as e:
            print(f"Error processing {image_file}: {e}")

# ==========================================
# Configuration and Execution
# ==========================================
if __name__ == "__main__":
    # Define your paths here
    CSV_FILE_PATH = "./session_20260728_102908/yolo_features.csv"                  # Path to your CSV file
    INPUT_IMAGES_DIR = "./session_20260728_102908/images"               # Folder containing your original images
    OUTPUT_IMAGES_DIR = "./cropped_yolo"   # Folder where platform crops will be saved

    print("Starting platform cropping process...")
    crop_platform_from_csv(CSV_FILE_PATH, INPUT_IMAGES_DIR, OUTPUT_IMAGES_DIR)
    print("Process complete!")