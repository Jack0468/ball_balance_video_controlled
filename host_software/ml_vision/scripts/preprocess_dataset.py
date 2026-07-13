import cv2
import pandas as pd
import argparse
import sys
import os

class DatasetPreprocessor:
    def __init__(self, video_path, synced_csv_path, output_dir, crop_box):
        self.video_path = video_path
        self.synced_csv_path = synced_csv_path
        self.output_dir = output_dir
        
        # crop_box is expected as "x,y,w,h" (output from select_crop.py)
        try:
            cx, cy, cw, ch = map(int, crop_box.split(','))
            self.crop_x1 = cx
            self.crop_y1 = cy
            self.crop_x2 = cx + cw
            self.crop_y2 = cy + ch
        except Exception:
            print("Error parsing crop box. Must be 'x,y,w,h'.")
            sys.exit(1)
            
    def run_preprocessing(self):
        print(f"Loading synced mapping from {self.synced_csv_path}...")
        df = pd.read_csv(self.synced_csv_path)
        
        # Prepare output directories
        images_dir = os.path.join(self.output_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)
        
        print(f"Opening video {self.video_path}...")
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            sys.exit(1)
            
        print("Extracting, cropping, and saving images...")
        processed_data = []
        
        for i, row in df.iterrows():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_idx = int(row['frame_index'])
            host_ms = int(row['host_timestamp_ms'])
            
            # Crop the frame
            cropped = frame[self.crop_y1:self.crop_y2, self.crop_x1:self.crop_x2]
            
            # Resize to standardized 640x480
            resized = cv2.resize(cropped, (640, 480))
            
            # Save image
            img_filename = f"{host_ms}.jpg"
            img_path = os.path.join(images_dir, img_filename)
            cv2.imwrite(img_path, resized)
            
            # Create a new row for the final labels.csv
            new_row = row.copy()
            new_row['image_file'] = img_filename
            processed_data.append(new_row)
            
            if i % 1000 == 0 and i > 0:
                print(f"Processed {i} frames...")
                
        cap.release()
        
        # Save final labels.csv
        labels_path = os.path.join(self.output_dir, 'labels.csv')
        df_final = pd.DataFrame(processed_data)
        
        # Drop intermediate sync columns we don't need in final labels
        if 'frame_index' in df_final.columns:
            df_final.drop(columns=['frame_index'], inplace=True)
        if 'video_time_ms' in df_final.columns:
            df_final.drop(columns=['video_time_ms'], inplace=True)
            
        # Move image_file to first column
        cols = ['image_file'] + [col for col in df_final.columns if col != 'image_file']
        df_final = df_final[cols]
        
        # Append mode?
        header = not os.path.exists(labels_path)
        df_final.to_csv(labels_path, mode='a', header=header, index=False)
        
        print(f"Successfully finished preprocessing. Labels appended to {labels_path}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess synchronized dataset (crop, resize, save).")
    parser.add_argument('--video', required=True, help="Path to raw .MOV video")
    parser.add_argument('--synced-csv', required=True, help="Path to intermediate synced telemetry mapping")
    parser.add_argument('--output-dir', default="host_software/ml_vision/data/02_silver", help="Output directory")
    parser.add_argument('--crop', required=True, help="Crop box as 'x,y,w,h' e.g. '82,435,915,762'")
    
    args = parser.parse_args()
    
    preprocessor = DatasetPreprocessor(args.video, args.synced_csv, args.output_dir, args.crop)
    preprocessor.run_preprocessing()

if __name__ == "__main__":
    main()
