import os
import pandas as pd
import json
import cv2
import glob

def generate_vla_dataset(bronze_dir, gold_dir):
    """
    Extracts frames from all session RGB videos in 01_bronze, pairs them with 
    synchronous telemetry, synthesizes the language command, and outputs a 
    unified VLA dataset ready for PyTorch.
    """
    images_dir = os.path.join(gold_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    vla_dataset = []
    
    # Find all session directories in bronze_dir
    session_dirs = glob.glob(os.path.join(bronze_dir, "session_*"))
    
    if not session_dirs:
        print(f"No session directories found in {bronze_dir}")
        return
        
    for session_dir in session_dirs:
        session_name = os.path.basename(session_dir)
        csv_path = os.path.join(session_dir, 'telemetry.csv')
        video_path = os.path.join(session_dir, 'rgb_video.mp4')
        
        if not os.path.exists(csv_path) or not os.path.exists(video_path):
            print(f"Skipping {session_name}: Missing telemetry or video.")
            continue
            
        print(f"Processing {session_name}...")
        df = pd.read_csv(csv_path)
        cap = cv2.VideoCapture(video_path)
        
        for idx, row in df.iterrows():
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: Video stream ended before telemetry for {session_name}")
                break
                
            frame_idx = int(row['frame_index'])
            
            # Save frame image
            image_name = f"{session_name}_frame_{frame_idx:05d}.jpg"
            image_path = os.path.join(images_dir, image_name)
            cv2.imwrite(image_path, frame)
            
            # Synthesize the audio command string based on the target coordinates.
            tx, ty = row['target_x'], row['target_y']
            audio_command = "hold"
            if tx > 30: audio_command = "go red"
            elif tx < -30: audio_command = "go blue"
            elif ty > 30: audio_command = "go green"
            elif ty < -30: audio_command = "go yellow"
            
            sample = {
                "timestamp_ms": row['host_timestamp_ms'],
                "image_path": os.path.abspath(image_path), # Use absolute for easier loading
                "audio_command": audio_command,
                "state_x": row['touch_x'],
                "state_y": row['touch_y'],
                "action_theta_a": row['theta_a'],
                "action_theta_b": row['theta_b'],
                "action_theta_c": row['theta_c']
            }
            vla_dataset.append(sample)
            
        cap.release()
        
    output_file = os.path.join(gold_dir, "vla_dataset.json")
    with open(output_file, 'w') as f:
        json.dump(vla_dataset, f, indent=2)
        
    print(f"Generated VLA dataset with {len(vla_dataset)} samples at {output_file}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bronze_data_dir = os.path.abspath(os.path.join(script_dir, "../../data/01_bronze"))
    gold_dir = os.path.abspath(os.path.join(script_dir, "../../data/03_gold"))
    
    generate_vla_dataset(bronze_data_dir, gold_dir)
