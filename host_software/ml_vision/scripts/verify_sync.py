import cv2
import pandas as pd
import os

print("Loading labels.csv...")
df = pd.read_csv('host_software/ml_vision/data/02_silver/labels.csv')

# Find a segment where the ball is moving
start_idx = 3000
end_idx = start_idx + 150

print(f"Creating video for frames {start_idx} to {end_idx}...")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_path = os.path.join('host_software/ml_vision/data/02_silver/sync_verification.mp4')
out = cv2.VideoWriter(out_path, fourcc, 30.0, (640, 480))

for i in range(start_idx, end_idx):
    if i >= len(df):
        break
    row = df.iloc[i]
    img_path = os.path.join('host_software/ml_vision/data/02_silver/images', str(row['image_file']))
    
    if not os.path.exists(img_path):
        continue
        
    frame = cv2.imread(img_path)
    
    tx = float(row['touch_x'])
    ty = float(row['touch_y'])
    
    px = int(320 + (tx / 140.0) * 640)
    py = int(240 - (ty / 110.0) * 480)
    
    cv2.circle(frame, (px, py), 15, (0, 0, 255), 3)
    cv2.putText(frame, f"X:{tx:.1f} Y:{ty:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    
    out.write(frame)

out.release()
print(f"Saved verification video to {out_path}")
