import os
import torch
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample


def generate_noise_profile(wav_path, out_path, target_sr=16000):
    if not os.path.exists(wav_path):
        print(f"Error: Could not find {wav_path}")
        print("Please convert your .m4a file to .wav and place it at this location.")
        return False

    print(f"Loading {wav_path}...")
    sr, data = wavfile.read(wav_path)

    # Convert to float32
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0

    # Convert to mono if stereo
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    # Resample if needed
    if sr != target_sr:
        print(f"Resampling from {sr}Hz to {target_sr}Hz...")
        num_samples = int(len(data) * target_sr / sr)
        data = resample(data, num_samples)

    # Create tensor
    waveform = torch.tensor(data, dtype=torch.float32)
    # Shape must be [Batch, Time] for our STFT logic
    waveform = waveform.unsqueeze(0)

    # Compute STFT (Matching our AudioCommandReceiver precisely)
    print("Computing STFT...")
    window = torch.hann_window(255)
    stft = torch.stft(
        waveform,
        n_fft=255,
        hop_length=128,
        win_length=255,
        window=window,
        center=False,
        return_complex=True,
    )

    # Get magnitude
    magnitude = stft.abs()

    # Compute average magnitude across the time dimension (dim=2)
    # Resulting shape will be [1, 128]
    noise_profile = magnitude.mean(dim=2)

    # Save the profile
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(noise_profile, out_path)

    print(f"Successfully generated noise profile with shape {noise_profile.shape}")
    print(f"Saved to: {out_path}")
    return True


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)

    wav_file = os.path.join(
        parent_dir, "data", "01_background_noise", "robot_background_sound.wav"
    )
    out_file = os.path.join(parent_dir, "models", "noise_profile.pt")

    generate_noise_profile(wav_file, out_file)
