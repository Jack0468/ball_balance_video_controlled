# ML Audio

This directory contains the audio processing pipeline and classification models for the Ball Balancing Robot. The primary model is a PyTorch-based neural network that runs continuously to listen for user voice commands (e.g., "go_red", "stop").

> [!IMPORTANT]
> **Data Management:** All audio datasets (raw recordings, cleaned clips, and ML-ready splits) must strictly follow the **Medallion Architecture (Bronze, Silver, Gold)** as defined in `docs/ENGINEERING_STANDARDS.md`.

## Features

- **`audio_receiver_pytorch.py`**: A threaded audio receiver that captures microphone data, calculates the Short-Time Fourier Transform (STFT), applies a noise profile, and feeds it into the PyTorch classifier.
- **`audio_command_classifier_pytorch.py`**: The neural network model definition.

## Export audio model weights for FPGA/HLS

Use `export_audio_weights.py` to generate a C header from the saved Keras model checkpoint:

```powershell
python .\export_audio_weights.py --hls
```

This creates `host_software/ml_audio/models/audio_command_classifier/audio_classifier_weights.h` with layer weights and biases exported in HLS-style arrays and shape macros.

If you want to export a different model path, pass `--model` and `--output` explicitly:

```powershell
python .\export_audio_weights.py --model .\models\audio_command_classifier\best_classifier.keras --output .\models\audio_command_classifier\audio_classifier_weights.h --hls
```
