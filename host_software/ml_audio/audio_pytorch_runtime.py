from pathlib import Path

import numpy as np
import torch

try:
    from .audio_command_classifier_pytorch import load_audio_command_classifier
except ImportError:
    from audio_command_classifier_pytorch import load_audio_command_classifier


SAMPLE_RATE = 16_000
MODEL_WINDOW_SECONDS = 1.25
OUTPUT_SEQUENCE_LENGTH = int(SAMPLE_RATE * MODEL_WINDOW_SECONDS)
FRAME_LENGTH = 255
FFT_LENGTH = 256
HOP_LENGTH = 128
LABEL_NAMES = ["go_blue", "go_green", "go_red", "go_yellow", "hold", "stop"]


def default_model_path():
    return Path(__file__).resolve().parent / "models" / "audio_command_classifier" / "pytorch" / "audio_command_classifier_state_dict.pth"


def align_speech_to_fixed_length(audio, target_samples=OUTPUT_SEQUENCE_LENGTH):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    peak = np.max(np.abs(audio))
    rms = np.sqrt(np.mean(audio ** 2))

    if peak < 0.03 or rms < 0.003:
        return None, {"reason": "too_quiet", "peak": float(peak), "rms": float(rms)}

    threshold = max(0.015, peak * 0.08)
    active = np.where(np.abs(audio) > threshold)[0]
    if len(active) == 0:
        return None, {"reason": "no_speech_detected"}

    start = max(0, active[0] - int(0.08 * SAMPLE_RATE))
    end = min(len(audio), active[-1] + int(0.12 * SAMPLE_RATE))
    audio = audio[start:end]

    if len(audio) > target_samples:
        audio = audio[:target_samples]
    if len(audio) < target_samples:
        audio = np.pad(audio, (0, target_samples - len(audio)))

    peak = np.max(np.abs(audio))
    if peak > 1e-6:
        audio = audio / peak * 0.95

    return audio.astype(np.float32), {"reason": "ok"}


def waveform_to_spectrogram(waveform):
    waveform_tensor = torch.as_tensor(waveform, dtype=torch.float32)
    if waveform_tensor.ndim == 1:
        waveform_tensor = waveform_tensor.unsqueeze(0)

    window = torch.hann_window(
        FRAME_LENGTH,
        periodic=True,
        dtype=waveform_tensor.dtype,
        device=waveform_tensor.device,
    )

    spec = torch.stft(
        waveform_tensor,
        n_fft=FFT_LENGTH,
        hop_length=HOP_LENGTH,
        win_length=FRAME_LENGTH,
        window=window,
        return_complex=True,
        center=False,
    )
    spec = spec.abs()
    spec = torch.log(spec + 1e-6)
    spec = spec.transpose(1, 2)
    return spec


def load_pytorch_audio_model(model_path=None, device=None):
    if model_path is None:
        model_path = default_model_path()
    model_path = Path(model_path)

    if device is None:
        device = torch.device("cpu")

    model = load_audio_command_classifier()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, model_path


def predict_probabilities(model, waveform, device=None):
    if device is None:
        device = next(model.parameters(), None)
        if device is None:
            device = next(model.buffers()).device
        else:
            device = device.device

    spec = waveform_to_spectrogram(waveform).to(device)
    with torch.no_grad():
        logits = model(spec)
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    return probs