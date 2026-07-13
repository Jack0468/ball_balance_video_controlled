import cv2
import pandas as pd
import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Quickly verify video/telemetry sync without full preprocessing.")
    parser.add_argument('--video', required=True, help="Path to raw .MOV video")
    parser.add_argument('--synced-csv', required=True, help="Path to synced_telemetry.csv")
    parser.add_argument('--crop', required=True, help="Crop box as 'x1,y1,x2,y2' or 'x,y,w,h'")
    parser.add_argument('--start-idx', type=int, default=3000, help="Frame index to start the 10-second preview (default: 3000)")
    parser.add_argument('--output', default='sync_check.mp4', help="Output mp4 path (default: sync_check.mp4)")
    args = parser.parse_args()

    print(f"Loading {args.synced_csv}...")
    df = pd.read_csv(args.synced_csv)
    
    # Parse crop string. Support both x1,y1,x2,y2 and x,y,w,h formats.
    # bounding_boxes_for_data.md has (x, y, w, h) but preprocess_dataset.py uses x1,y1,x2,y2
    # The user passes "82,435,915,762" which is x,y,w,h. Wait, preprocess_dataset.py does:
    # frame[y1:y2, x1:x2] where x2 = x + w and y2 = y + h.
    try:
        parts = list(map(int, args.crop.split(',')))
        cx, cy, cw, ch = parts
        x1 = cx
        y1 = cy
        x2 = cx + cw
        y2 = cy + ch
    except Exception:
        print("Error parsing crop box. Must be 'x,y,w,h'.")
        sys.exit(1)

    print(f"Opening video {args.video}...")
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("Error: Could not open video.")
        sys.exit(1)
        
    start_idx = args.start_idx
    end_idx = start_idx + 300 # 10 seconds at 30fps
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, 30.0, (640, 480))
    
    # Skip to start_idx
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
    
    print(f"Generating 10-second preview from frame {start_idx} to {end_idx}...")
    
    for i in range(start_idx, end_idx):
        if i >= len(df):
            break
            
        ret, frame = cap.read()
        if not ret:
            break
            
        # Get the corresponding telemetry row for this frame_index
        row = df[df['frame_index'] == i]
        if row.empty:
            continue
        row = row.iloc[0]
        
        # Crop & Resize
        cropped = frame[y1:y2, x1:x2]
        if cropped.size == 0:
            continue
        resized = cv2.resize(cropped, (640, 480))
        
        # Get telemetry touch coordinates
        tx = float(row['touch_x'])
        ty = float(row['touch_y'])
        
        # Map mm coordinates to 640x480 pixels, treating (0,0) as the exact center (320, 240)
        # Platform is 140mm wide (-70 to 70) and 110mm tall (-55 to 55)
        # In OpenCV, Y=0 is the TOP of the image. Assuming standard Cartesian coordinates 
        # where +Y is UP, we subtract the Y offset from the center.
        px = int(320 + (tx / 140.0) * 640)
        py = int(240 - (ty / 110.0) * 480)
        
        # Get telemetry target coordinates (the ball)
        bx = float(row['target_x'])
        by = float(row['target_y'])
        bpx = int(320 + (bx / 140.0) * 640)
        bpy = int(240 - (by / 110.0) * 480)
        
        # Draw Touch (Red)
        cv2.circle(resized, (px, py), 15, (0, 0, 255), 3)
        # Draw Ball Target (Green)
        cv2.circle(resized, (bpx, bpy), 15, (0, 255, 0), 3)
        
        cv2.putText(resized, f"Touch X:{tx:.1f} Y:{ty:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.putText(resized, f"Target X:{bx:.1f} Y:{by:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(resized, f"Frame: {i}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        out.write(resized)

    cap.release()
    out.release()
    print(f"Done! Saved verification video to {args.output}")

if __name__ == "__main__":
    main()
