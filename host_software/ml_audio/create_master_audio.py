import os
import glob
import numpy as np
import soundfile as sf

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bg_dir = os.path.join(script_dir, 'data', '01_background_noise')
    samples_dir = os.path.join(script_dir, 'data', '01_evaluation_samples')
    out_dir = os.path.join(script_dir, 'data', '02_silver') # Let's save it to 02_silver
    os.makedirs(out_dir, exist_ok=True)
    
    out_path = os.path.join(out_dir, 'master_evaluation_audio.wav')
    
    # 1. Load Background Noise
    bg_files = glob.glob(os.path.join(bg_dir, '*.wav'))
    if not bg_files:
        print(f"ERROR: No background noise .wav files found in {bg_dir}")
        print("Please place at least one background noise file there.")
        return
        
    print(f"Loading background noise from {bg_files[0]}")
    bg_audio, bg_sr = sf.read(bg_files[0])
    
    if bg_audio.ndim > 1:
        bg_audio = np.mean(bg_audio, axis=1) # mix to mono
        
    TARGET_SR = 16000
    if bg_sr != TARGET_SR:
        print(f"WARNING: Background sample rate is {bg_sr}, but 16000 is required.")
        # Very crude resample or assume it's acceptable (if user recorded at 16k)
        
    TOTAL_DURATION_SEC = 120
    TOTAL_SAMPLES = TOTAL_DURATION_SEC * TARGET_SR
    
    # Loop background noise to 120 seconds
    master_audio = np.zeros(TOTAL_SAMPLES, dtype=np.float32)
    repeats = int(np.ceil(TOTAL_SAMPLES / len(bg_audio)))
    looped_bg = np.tile(bg_audio, repeats)[:TOTAL_SAMPLES]
    master_audio += looped_bg
    
    # 2. Overlay commands
    EVAL_SEQUENCE = [
        ("go_grey", 0),
        ("go_blue", 10),
        ("go_green", 20),
        ("go_yellow", 30),
        ("go_red", 40),
        ("FORWARD", 50),
        ("LEFT", 60),
        ("RIGHT", 70),
        ("BACKWARD", 80),
        ("HOLD", 90),
        ("STOP", 100),
        ("_background_", 110)
    ]
    
    print("Overlaying commands...")
    for cmd_name, start_sec in EVAL_SEQUENCE:
        if cmd_name == "_background_":
            print(f"[{start_sec:03d}s] Leaving background as is for {cmd_name}")
            continue
            
        pattern = os.path.join(samples_dir, f"*{cmd_name}.wav")
        matches = glob.glob(pattern)
        if not matches:
            print(f"[{start_sec:03d}s] WARNING: No file matching {pattern} found! Skipping...")
            continue
            
        cmd_path = matches[0]
            
        cmd_audio, cmd_sr = sf.read(cmd_path)
        if cmd_audio.ndim > 1:
            cmd_audio = np.mean(cmd_audio, axis=1)
            
        start_sample = int(start_sec * TARGET_SR)
        end_sample = start_sample + len(cmd_audio)
        
        if end_sample > TOTAL_SAMPLES:
            cmd_audio = cmd_audio[:TOTAL_SAMPLES - start_sample]
            end_sample = TOTAL_SAMPLES
            
        # Add (overlay) the command onto the background noise
        print(f"[{start_sec:03d}s] Overlaying {cmd_name}.wav")
        master_audio[start_sample:end_sample] += cmd_audio
        
    # Normalize to prevent clipping
    peak = np.max(np.abs(master_audio))
    if peak > 1.0:
        print(f"Normalizing to prevent clipping (Peak was {peak:.2f})")
        master_audio = master_audio / peak * 0.95
        
    sf.write(out_path, master_audio, TARGET_SR)
    print(f"\nSuccessfully generated master audio file: {out_path}")
    print(f"Duration: {TOTAL_DURATION_SEC} seconds, Sample Rate: {TARGET_SR} Hz")

if __name__ == '__main__':
    main()
