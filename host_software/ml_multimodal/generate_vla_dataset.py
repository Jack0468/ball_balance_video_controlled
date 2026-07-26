import os
import pandas as pd
import numpy as np
import json

def generate_vla_dataset(silver_dir, output_dir):
    """
    Temporally aligns Vision (images), Audio (commands derived from targets), 
    and Telemetry (theta_a, theta_b, theta_c) into a unified dataset for VLA Imitation Learning.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(silver_dir, 'labels_sequential.csv')
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    
    # We want to create a JSON lines (or simple JSON list) describing the VLA tokens.
    # VLA Input: Image path, Audio command (simulated from target_x/y), current state (touch_x, touch_y)
    # VLA Output: theta_a, theta_b, theta_c
    
    vla_dataset = []
    
    for idx, row in df.iterrows():
        # Synthesize the audio command string based on the target coordinates.
        # In reality, this would be a wav2vec embedding or transcribed text.
        tx, ty = row['target_x'], row['target_y']
        audio_command = "hold"
        if tx > 30: audio_command = "go red"
        elif tx < -30: audio_command = "go blue"
        elif ty > 30: audio_command = "go green"
        elif ty < -30: audio_command = "go yellow"
        
        sample = {
            "timestamp_ms": row['host_timestamp_ms'],
            "image_path": os.path.join(silver_dir, "images", row['image_file']),
            "audio_command": audio_command,
            "state_x": row['touch_x'],
            "state_y": row['touch_y'],
            "action_theta_a": row['theta_a'],
            "action_theta_b": row['theta_b'],
            "action_theta_c": row['theta_c']
        }
        vla_dataset.append(sample)
        
    output_file = os.path.join(output_dir, "vla_dataset.json")
    with open(output_file, 'w') as f:
        json.dump(vla_dataset, f, indent=2)
        
    print(f"Generated VLA dataset with {len(vla_dataset)} samples at {output_file}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    silver_data_dir = os.path.abspath(os.path.join(script_dir, "../ml_vision/data/02_silver"))
    out_dir = os.path.abspath(os.path.join(script_dir, "data/03_gold"))
    
    generate_vla_dataset(silver_data_dir, out_dir)
