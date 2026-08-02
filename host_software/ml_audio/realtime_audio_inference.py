import sys
import os
import time
import numpy as np

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
        script_dir, "synthetic", "models", "pytorch", "audio_weights_with_synthetic.pth"
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

    command_history = ["hold"]

    try:
        while True:
            command = receiver.get_latest_command()
            if command:
                # If the command is already in the history, remove it so it can be moved to the front
                if command in command_history:
                    command_history.remove(command)

                command_history.append(command)

                # Keep only the last 5 unique commands
                if len(command_history) > 5:
                    command_history.pop(0)

            # Grab current volume for diagnostic
            vol = np.max(np.abs(receiver.audio_buffer))

            # Format the history for display
            history_str = " <- ".join(reversed(command_history))

            # Print continuously with carriage return to update in place
            sys.stdout.write(f"\r[AUDIO] Vol: {vol:.3f} | History: {history_str:<60}")
            sys.stdout.flush()

            # Poll at a high frequency so it feels instantaneous
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nStopping audio receiver...")
    finally:
        receiver.stop()


if __name__ == "__main__":
    main()
