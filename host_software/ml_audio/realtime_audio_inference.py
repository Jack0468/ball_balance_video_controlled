import sys
import os
import time

# Ensure we can import from ml_audio even if run from this directory directly
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from ml_audio.audio_receiver_pytorch import AudioCommandReceiver
import sounddevice as sd

def main():
    # Construct the path to the PyTorch model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(
        script_dir, 
        'models', 
        'audio_command_classifier', 
        'pytorch', 
        'audio_command_classifier_state_dict.pth'
    )
    
    print("Initializing real-time audio receiver (PyTorch)...")
    
    # Display microphone diagnostic info
    default_input_idx = sd.default.device[0]
    device_info = sd.query_devices(default_input_idx)
    sd.default.samplerate = 16000  # Enforce 16 kHz globally for sounddevice
    print(f"\n--- Microphone Diagnostic ---")
    print(f"Device ID: {default_input_idx}")
    print(f"Name: {device_info['name']}")
    print(f"Hardware Default Sample Rate: {device_info['default_samplerate']} Hz")
    print(f"Script Requested Sample Rate: 16000 Hz")
    print(f"Channels: {device_info['max_input_channels']}")
    print(f"-----------------------------\n")
    
    receiver = AudioCommandReceiver(model_path)
    
    print("\nListening for commands continuously... (Press Ctrl+C to quit)\n")
    
    current_command = "hold"
    
    try:
        while True:
            command = receiver.get_latest_command()
            if command:
                current_command = command
            
            # Print continuously with carriage return to update in place
            sys.stdout.write(f"\r[AUDIO] Current command: {current_command:<15}")
            sys.stdout.flush()
            
            # Poll at a high frequency so it feels instantaneous
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\nStopping audio receiver...")
    finally:
        receiver.stop()

if __name__ == "__main__":
    main()
