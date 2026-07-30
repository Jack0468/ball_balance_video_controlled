import cv2
from PIL import Image
import sys
import os

def video_to_gif(video_path, gif_path, max_frames=50, skip_frames=2, resize_width=480):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open {video_path}")
        return
        
    frames = []
    frame_count = 0
    saved_frames = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % skip_frames == 0:
            h, w = frame.shape[:2]
            new_h = int(h * (resize_width / w))
            frame_resized = cv2.resize(frame, (resize_width, new_h))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
            saved_frames += 1
            
            if saved_frames >= max_frames:
                break
                
        frame_count += 1
        
    cap.release()
    
    if frames:
        os.makedirs(os.path.dirname(gif_path), exist_ok=True)
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            optimize=True,
            duration=int((1000/30)*skip_frames),
            loop=0
        )
        print(f"Saved {gif_path}")

video_to_gif(r'C:\Users\Admin\Documents\Windows_codespace\VRI_2026\host_software\data\01_bronze\session_20260728_102908\rgb_video.mp4', r'C:\Users\Admin\Documents\Windows_codespace\VRI_2026\docs\assets\rgb_video_demo.gif', max_frames=60, skip_frames=2)
video_to_gif(r'C:\Users\Admin\Documents\Windows_codespace\VRI_2026\host_software\data\01_bronze\video1\sync_check.mp4', r'C:\Users\Admin\Documents\Windows_codespace\VRI_2026\docs\assets\sync_check_demo.gif', max_frames=60, skip_frames=2)
