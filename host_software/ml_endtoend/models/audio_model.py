import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# STFT Constants matching the trained audio pipeline
N_FFT = 256
WIN_LENGTH = 255
HOP_LENGTH = 128
BN_EPS = 1e-3
NORM_EPS = 1e-7


class AudioModel(nn.Module):
    """
    Audio Command Classifier network matching the PyTorch state_dict format.
    Accepts either raw waveform audio tensors [B, num_samples] or pre-computed 
    spectrograms [B, 1, H, W].
    """

    def __init__(self, num_classes=5):
        super().__init__()
        self.num_classes = num_classes
        self.norm_epsilon = NORM_EPS
        self.bn1_eps = BN_EPS
        self.bn2_eps = BN_EPS
        self.bn3_eps = BN_EPS

        # Hann window buffer for PyTorch STFT
        self.register_buffer("stft_window", torch.hann_window(WIN_LENGTH, periodic=True), persistent=False)

        # Registered buffers matching the state_dict key layout
        self.register_buffer("norm_mean", torch.zeros(1))
        self.register_buffer("norm_variance", torch.ones(1))

        self.register_buffer("conv1_weight", torch.zeros(12, 1, 3, 3))
        self.register_buffer("conv1_bias", torch.zeros(12))
        self.register_buffer("bn1_gamma", torch.ones(12))
        self.register_buffer("bn1_beta", torch.zeros(12))
        self.register_buffer("bn1_mean", torch.zeros(12))
        self.register_buffer("bn1_var", torch.ones(12))

        self.register_buffer("conv2_weight", torch.zeros(24, 12, 3, 3))
        self.register_buffer("conv2_bias", torch.zeros(24))
        self.register_buffer("bn2_gamma", torch.ones(24))
        self.register_buffer("bn2_beta", torch.zeros(24))
        self.register_buffer("bn2_mean", torch.zeros(24))
        self.register_buffer("bn2_var", torch.ones(24))

        self.register_buffer("conv3_weight", torch.zeros(48, 24, 3, 3))
        self.register_buffer("conv3_bias", torch.zeros(48))
        self.register_buffer("bn3_gamma", torch.ones(48))
        self.register_buffer("bn3_beta", torch.zeros(48))
        self.register_buffer("bn3_mean", torch.zeros(48))
        self.register_buffer("bn3_var", torch.ones(48))

        self.register_buffer("dense_weight", torch.zeros(num_classes, 48))
        self.register_buffer("dense_bias", torch.zeros(num_classes))

    def waveform_to_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """Converts raw waveform audio [B, S] to log-magnitude STFT spectrogram [B, 1, T, F]."""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Compute STFT
        spec = torch.stft(
            waveform,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            window=self.stft_window,
            center=False,
            return_complex=True,
        )
        # Log magnitude
        spec = torch.log(spec.abs() + 1e-6)
        # Reshape to [B, 1, time_steps, freq_bins]
        spec = spec.transpose(1, 2).unsqueeze(1)
        return spec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If 1D/2D audio waveform is passed, run STFT first
        if x.dim() <= 2:
            x = self.waveform_to_spectrogram(x)
        elif x.dim() == 3:
            x = x.unsqueeze(1)

        # Resize to expected model input dimensions (64, 64)
        if x.shape[-2:] != (64, 64):
            x = F.interpolate(x, size=(64, 64), mode="bilinear", align_corners=False)

        # Input Normalization
        x = (x - self.norm_mean.view(1, -1, 1, 1)) / torch.sqrt(
            self.norm_variance.view(1, -1, 1, 1) + self.norm_epsilon
        )

        # Conv Layer 1
        x = F.conv2d(x, self.conv1_weight, bias=self.conv1_bias, padding=1)
        x = F.relu(x)
        x = F.batch_norm(
            x,
            self.bn1_mean,
            self.bn1_var,
            self.bn1_gamma,
            self.bn1_beta,
            training=False,
            eps=self.bn1_eps,
        )
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # Conv Layer 2
        x = F.conv2d(x, self.conv2_weight, bias=self.conv2_bias, padding=1)
        x = F.relu(x)
        x = F.batch_norm(
            x,
            self.bn2_mean,
            self.bn2_var,
            self.bn2_gamma,
            self.bn2_beta,
            training=False,
            eps=self.bn2_eps,
        )
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # Conv Layer 3
        x = F.conv2d(x, self.conv3_weight, bias=self.conv3_bias, padding=1)
        x = F.relu(x)
        x = F.batch_norm(
            x,
            self.bn3_mean,
            self.bn3_var,
            self.bn3_gamma,
            self.bn3_beta,
            training=False,
            eps=self.bn3_eps,
        )

        # Global Average Pooling & Linear Output
        x = x.mean(dim=(2, 3))
        logits = F.linear(x, self.dense_weight, self.dense_bias)
        return logits


def load_audio_model(weights_path: str = "audio_model_weights.pth") -> AudioModel:
    """Helper function to load the weights state_dict into the AudioModel."""
    weights_p = Path(weights_path)
    labels_path = weights_p.with_name("labels.json")

    # Read class count from labels.json if present
    num_classes = 5
    if labels_path.is_file():
        labels = json.loads(labels_path.read_text(encoding="utf-8"))
        num_classes = len(labels)

    state = torch.load(weights_p, map_location="cpu", weights_only=True)
    if isinstance(state, dict):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
        if "dense_bias" in state:
            num_classes = state["dense_bias"].shape[0]

    model = AudioModel(num_classes=num_classes)
    
    if isinstance(state, dict):
        model.load_state_dict(state, strict=True)
    model.eval()
    return model


if __name__ == "__main__":
    # Test instantiation and forward pass with dummy 1.25s waveform (20,000 samples @ 16kHz)
    dummy_waveform = torch.randn(1, 20000)
    model = AudioModel(num_classes=5)
    output_logits = model(dummy_waveform)
    print(f"AudioModel initialized successfully. Output shape: {output_logits.shape}")