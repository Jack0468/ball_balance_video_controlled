import os
from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()

def convert_heic_to_jpg(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    labels_file = os.path.join(input_dir, "labels.txt")
    if not os.path.exists(labels_file):
        print(f"Error: {labels_file} not found.")
        return

    # Read labels
    with open(labels_file, "r") as f:
        lines = f.readlines()[1:] # Skip header
        
    # Copy and adapt labels
    new_labels_file = os.path.join(output_dir, "labels.txt")
    with open(new_labels_file, "w") as f:
        f.write("filename,x1,y1,x2,y2,x3,y3,x4,y4\n")
        
        for line in lines:
            parts = line.strip().split(",")
            filename_base = parts[0]
            coords = parts[1:]
            
            # Find the exact match or timestamped match
            heic_path = os.path.join(input_dir, f"{filename_base}.HEIC")
            if not os.path.exists(heic_path):
                for file in os.listdir(input_dir):
                    if file.startswith(filename_base) and file.endswith(".HEIC"):
                        heic_path = os.path.join(input_dir, file)
                        break
            
            if os.path.exists(heic_path):
                img = Image.open(heic_path)
                jpg_filename = f"{filename_base}.jpg"
                jpg_path = os.path.join(output_dir, jpg_filename)
                
                img = img.convert("RGB")
                img.save(jpg_path, "JPEG")
                
                print(f"Converted {heic_path} -> {jpg_path}")
                
                # Write to new labels.txt
                f.write(f"{jpg_filename},{','.join(coords)}\n")
            else:
                print(f"Warning: Could not find HEIC file for {filename_base}")

if __name__ == "__main__":
    input_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/image/images_find_plane1"))
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/image/base_images"))
    convert_heic_to_jpg(input_dir, output_dir)
    print("Done converting HEIC to JPG.")
