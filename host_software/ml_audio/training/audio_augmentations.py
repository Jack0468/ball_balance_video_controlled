"""Waveform-level augmentations for audio command training.

Pure numpy/scipy -- no torchaudio/librosa/audiomentations available in the
project env (checked: only soundfile + scipy are installed) and none of
these techniques need more than that. Keeping it dependency-light also means
the eventual Colab script only needs to `pip install` what's already in
requirements.txt plus torch, not a heavier audio stack.

Each function takes and returns a 1D float32 numpy waveform of fixed length
(OUTPUT_SEQUENCE_LENGTH). Applied *before* the STFT, at the waveform level,
since that's the only place noise-mixing/speed-perturbation/reverb are
physically meaningful.
"""

import numpy as np
from scipy.signal import fftconvolve, resample

from ml_audio.evaluations.evaluate_audio_classifier import pad_or_truncate


def speed_perturb(waveform: np.ndarray, rate: float, target_samples: int) -> np.ndarray:
    """Resample by `rate` (e.g. 0.9-1.1), then re-fit to target_samples.
    Changes tempo and pitch together -- the standard "speed perturbation"
    augmentation (Ko et al. 2015), well established for speech robustness
    and simple to implement without a phase vocoder."""
    new_len = max(1, int(round(len(waveform) / rate)))
    resampled = resample(waveform, new_len).astype(np.float32)
    return pad_or_truncate(resampled, target_samples)


def noise_mix(waveform: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Mix in a same-length noise clip at a target SNR (dB)."""
    sig_rms = np.sqrt(np.mean(waveform**2)) + 1e-8
    noise_rms = np.sqrt(np.mean(noise**2)) + 1e-8
    target_noise_rms = sig_rms / (10 ** (snr_db / 20))
    scaled_noise = noise * (target_noise_rms / noise_rms)
    mixed = waveform + scaled_noise
    peak = np.max(np.abs(mixed))
    if peak > 1.0:
        mixed = mixed / peak
    return mixed.astype(np.float32)


def synthetic_reverb(waveform: np.ndarray, decay_tau: float, ir_len: int, rng: np.random.Generator) -> np.ndarray:
    """Convolve with a synthetic exponentially-decaying noise impulse
    response -- a cheap stand-in for a real room-impulse-response dataset
    (none available locally), enough to simulate "not a dry close-mic
    recording" for augmentation purposes."""
    t = np.arange(ir_len)
    ir = rng.standard_normal(ir_len).astype(np.float32) * np.exp(-t / decay_tau).astype(np.float32)
    ir /= np.sqrt(np.sum(ir**2)) + 1e-8
    wet = fftconvolve(waveform, ir, mode="full")[: len(waveform)]
    peak_in = np.max(np.abs(waveform)) + 1e-8
    peak_out = np.max(np.abs(wet)) + 1e-8
    return (wet * (peak_in / peak_out)).astype(np.float32)


def gain_jitter(waveform: np.ndarray, gain_db: float) -> np.ndarray:
    return (waveform * (10 ** (gain_db / 20))).astype(np.float32)


def time_shift(waveform: np.ndarray, shift_samples: int) -> np.ndarray:
    shifted = np.zeros_like(waveform)
    if shift_samples > 0:
        shifted[shift_samples:] = waveform[: len(waveform) - shift_samples]
    elif shift_samples < 0:
        shifted[: len(waveform) + shift_samples] = waveform[-shift_samples:]
    else:
        shifted[:] = waveform
    return shifted


class AugmentationConfig:
    """Which techniques are active and how often each fires per sample per
    epoch (probability 1.0 for a single-technique ablation run; <1.0 so a
    "combined" run doesn't stack every distortion on every sample)."""

    def __init__(
        self,
        name: str,
        use_noise: bool = False,
        use_speed: bool = False,
        use_reverb: bool = False,
        use_gain: bool = False,
        use_shift: bool = False,
        prob: float = 1.0,
    ):
        self.name = name
        self.use_noise = use_noise
        self.use_speed = use_speed
        self.use_reverb = use_reverb
        self.use_gain = use_gain
        self.use_shift = use_shift
        self.prob = prob

    def apply(
        self,
        waveform: np.ndarray,
        target_samples: int,
        noise_pool: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        out = waveform
        if self.use_speed and rng.random() < self.prob:
            rate = rng.uniform(0.9, 1.1)
            out = speed_perturb(out, rate, target_samples)
        if self.use_noise and rng.random() < self.prob:
            noise_idx = rng.integers(0, len(noise_pool))
            snr_db = rng.uniform(0.0, 15.0)
            out = noise_mix(out, noise_pool[noise_idx], snr_db)
        if self.use_reverb and rng.random() < self.prob:
            decay_tau = rng.uniform(800.0, 3000.0)
            out = synthetic_reverb(out, decay_tau, ir_len=4000, rng=rng)
        if self.use_gain and rng.random() < self.prob:
            out = gain_jitter(out, rng.uniform(-6.0, 6.0))
        if self.use_shift and rng.random() < self.prob:
            out = time_shift(out, int(rng.integers(-800, 800)))
        return out


# Named presets for production training (train_audio_command_classifier.py
# --augmentation) and for reference from the ablation script. Deliberately
# excludes reverb: the local ablation (see
# docs/plans/audio_eval_notebook_refactor_plan.md) found synthetic_reverb a
# clear, large regression (best val_acc 57.3% vs. 76.5% baseline at matched
# budget) -- not included here pending a better impulse-response source or a
# much gentler decay range.
PRESETS: dict[str, AugmentationConfig] = {
    "none": AugmentationConfig("none"),
    "light": AugmentationConfig("light", use_gain=True, use_shift=True),
    "speed": AugmentationConfig("speed", use_speed=True),
    "noise": AugmentationConfig("noise", use_noise=True),
    "combined": AugmentationConfig(
        "combined", use_noise=True, use_speed=True,
        use_gain=True, use_shift=True, prob=0.5,
    ),
}
