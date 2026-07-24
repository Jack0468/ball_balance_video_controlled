import os
import sys
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from audio_command_classifier_pytorch import AudioCommandClassifier

SAMPLE_RATE = 16000
CLIP_SECONDS = 1.25
TARGET_SAMPLES = int(SAMPLE_RATE * CLIP_SECONDS)

path = BASE_DIR / "data" / "bronze" / "audio" / "raw" / "speaker02" / "go_green" / "speaker02__go_green__001.wav"
if not path.exists():
    raise FileNotFoundError(path)

waveform, sr = sf.read(str(path), dtype="float32")
if sr != SAMPLE_RATE:
    raise ValueError(f"Expected {SAMPLE_RATE} Hz, got {sr}")
if waveform.ndim > 1:
    waveform = np.mean(waveform, axis=1)

peak = np.max(np.abs(waveform))
rms = np.sqrt(np.mean(waveform ** 2))
print("waveform shape", waveform.shape, "peak", peak, "rms", rms)

threshold = max(0.015, peak * 0.08)
active = np.where(np.abs(waveform) > threshold)[0]
print("threshold", threshold, "active count", len(active))
if len(active) == 0:
    raise ValueError("No active speech region found")

start = max(0, active[0] - int(0.08 * SAMPLE_RATE))
end = min(len(waveform), active[-1] + int(0.12 * SAMPLE_RATE))
waveform = waveform[start:end]
if len(waveform) > TARGET_SAMPLES:
    waveform = waveform[:TARGET_SAMPLES]
else:
    waveform = np.pad(waveform, (0, TARGET_SAMPLES - len(waveform)))

waveform = waveform / np.max(np.abs(waveform)) * 0.95
print("trimmed shape", waveform.shape)

waveform_t = torch.from_numpy(waveform)
window = torch.hann_window(512, periodic=False)
spec = torch.stft(
    waveform_t,
    n_fft=512,
    hop_length=128,
    win_length=512,
    window=window,
    center=True,
    return_complex=True,
    pad_mode="reflect",
)
spec = torch.abs(spec)
spec = torch.log(spec + 1e-6)
spec = spec.transpose(0, 1).unsqueeze(0)
print("spec shape", spec.shape, "min", spec.min().item(), "max", spec.max().item())

model = AudioCommandClassifier()
model.eval()
with torch.no_grad():
    logits = model(spec)
    probs = torch.softmax(logits, dim=-1)[0].numpy()
print("logits", logits.numpy())
print("probs", probs)
print("top index", int(np.argmax(probs)))
print("label_names", ["go_blue", "go_green", "go_red", "go_yellow", "hold", "stop"])
