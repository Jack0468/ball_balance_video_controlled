"""Pure DSP constants and the waveform->spectrogram transform, with zero
hardware/runtime dependencies (no sounddevice, no AudioCommandClassifier's
14000-line weight-literal file).

Split out of audio_receiver_pytorch.py so that training/evaluation code
(train_audio_command_classifier.py, evaluate_audio_classifier.py,
segment_background_recording.py) doesn't have to import sounddevice just to
get SAMPLE_RATE or run an STFT -- that matters most for portability to a
fresh environment (e.g. Colab) where sounddevice needs system-level
PortAudio libs that have nothing to do with training. audio_receiver_pytorch.py
imports these values from here rather than redefining them, so there's one
definition of the transform shared by the live receiver and every offline
script that needs to match it exactly.
"""

import torch

SAMPLE_RATE = 16_000
MODEL_WINDOW_SECONDS = 1.25
OUTPUT_SEQUENCE_LENGTH = int(SAMPLE_RATE * MODEL_WINDOW_SECONDS)
N_FFT = 255
HOP_LENGTH = 128


def waveform_to_spectrogram(waveform, noise_profile=None, noise_alpha=1.5):
    """Accepts either a single waveform (T,) or an already-batched (B, T)
    tensor/array -- the latter lets training code run one vectorized STFT
    over a whole minibatch instead of looping per-sample, which matters for
    actually using a GPU (this model is tiny; a Python per-sample loop
    dominates wall-clock and leaves the GPU idle regardless of device)."""
    waveform_pt = torch.as_tensor(waveform, dtype=torch.float32)
    if waveform_pt.dim() == 1:
        waveform_pt = waveform_pt.unsqueeze(0)
    window = torch.hann_window(N_FFT, device=waveform_pt.device)
    spec = torch.stft(
        waveform_pt,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        window=window,
        return_complex=True,
        center=False,
    )

    # Convert to absolute magnitude spectrum
    spec = spec.abs()

    # Apply Spectral Subtraction if profile exists
    if noise_profile is not None:
        # noise_profile is [1, 128], we need [1, 128, 1] to broadcast over the time dimension
        spec = spec - (noise_profile.unsqueeze(-1) * noise_alpha)
        spec = torch.clamp(spec, min=0.0)

    # PyTorch stft returns [batch, freqs, frames]. TF returns [batch, frames, freqs].
    spec = spec.transpose(1, 2)
    spec = torch.log(spec + 1e-6)
    return spec
