# ML Audio Module

This directory contains the audio processing pipeline and classification models for the Ball Balancing Robot.

> [!IMPORTANT]
> **Data Management:** All audio datasets (raw recordings, cleaned clips, and ML-ready splits) must strictly follow the **Medallion Architecture (Bronze, Silver, Gold)** as defined in `docs/ENGINEERING_STANDARDS.md`.

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
