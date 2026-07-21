import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import argparse

def generate_missing_ball_grid(data_dir, output_path):
    csv_path = os.path.join(data_dir, 'labels.csv')
    images_dir = os.path.join(data_dir, 'images')
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
        
    print(f"Reading raw telemetry from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # The physical screen is 167mm x 135.5mm (from Screen.cpp)
    # So max limits from center (0,0) are roughly X: +/- 83.5, Y: +/- 67.75
    # The ball has a radius of roughly 15-20mm. 
    # If the coordinate is > 75mm (X) or > 60mm (Y), the ball is definitively falling off the edge.
    
    OOB_X = 75.0
    OOB_Y = 60.0
    
    df_oob = df[(abs(df['touch_x']) > OOB_X) | (abs(df['touch_y']) > OOB_Y)]
    print(f"Found {len(df_oob)} telemetry rows where the ball is physically out of bounds.")
    
    # We want unique images (since labels.csv has duplicates)
    df_oob_unique = df_oob.drop_duplicates(subset=['image_file']).copy()
    print(f"Found {len(df_oob_unique)} unique image frames out of bounds.")
    
    if len(df_oob_unique) == 0:
        print("No out-of-bounds frames found!")
        return
        
    # Pick up to 16 images for a 4x4 grid
    num_samples = min(16, len(df_oob_unique))
    samples = df_oob_unique.sample(n=num_samples, random_state=42)
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    fig.suptitle("Out-of-Bounds (Missing Ball) Telemetry Verification", fontsize=16)
    
    for idx, (index, row) in enumerate(samples.iterrows()):
        ax = axes[idx // 4, idx % 4]
        img_path = os.path.join(images_dir, row['image_file'])
        
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img)
            ax.set_title(f"X: {row['touch_x']:.1f}, Y: {row['touch_y']:.1f}")
        else:
            ax.set_title("Image missing")
            
        ax.axis('off')
        
    # Hide any unused subplots
    for i in range(num_samples, 16):
        axes[i // 4, i % 4].axis('off')
        
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved verification grid to: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Find and visualize out-of-bounds frames")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()
    
    generate_missing_ball_grid(args.data_dir, args.output_path)
