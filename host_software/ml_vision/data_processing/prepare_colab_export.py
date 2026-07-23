import os
import zipfile
import yaml
import shutil

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, "../data"))
    
    export_dir = os.path.join(data_dir, "colab_export")
    os.makedirs(export_dir, exist_ok=True)
    
    # 1. Create the colab_dataset.yaml
    yaml_content = """path: /content/dataset
train:
  - yolo_raw_dataset/images
  - 03_synthetic_yolo/images
val:
  - yolo_raw_dataset/images

names:
  0: platform
  1: ball
  2: blue_marker
  3: grey_marker
  4: black_marker
  5: red_marker
  6: green_marker
  7: yellow_marker
  8: cyan_marker
  9: purple_marker
  10: orange_marker
  11: pink_marker
  12: brown_marker

# Keypoints (required for YOLOv8 Pose)
kpt_shape: [4, 3]
"""
    yaml_path = os.path.join(export_dir, "colab_dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
        
    print("Created colab_dataset.yaml")
    
    # 2. Zip everything up
    zip_path = os.path.join(data_dir, "colab_dataset.zip")
    print(f"Creating {zip_path} ... (this may take a minute)")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add YAML
        zipf.write(yaml_path, arcname="dataset/colab_dataset.yaml")
        
        # Add yolo_raw_dataset
        raw_dir = os.path.join(data_dir, "yolo_raw_dataset")
        for root, dirs, files in os.walk(raw_dir):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, data_dir)
                zipf.write(abs_path, arcname=f"dataset/{rel_path}")
                
        # Add 03_synthetic_yolo
        synth_dir = os.path.join(data_dir, "03_synthetic_yolo")
        for root, dirs, files in os.walk(synth_dir):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, data_dir)
                zipf.write(abs_path, arcname=f"dataset/{rel_path}")
                
    print(f"Done! You can now upload {zip_path} to Google Drive.")

if __name__ == '__main__':
    main()
