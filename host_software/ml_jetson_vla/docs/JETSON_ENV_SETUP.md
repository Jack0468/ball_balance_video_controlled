# Jetson AGX Orin Environment Setup (Track 1)

Bring-up notes for running `runtime/run_jetson_standalone.py` (small-class expert
pipeline, Phase A) on the Jetson AGX Orin 64GB Developer Kit (p3730). The repo's root
`environment.yml`/`requirements.txt` are x86 (laptop) oriented and **cannot be reused
verbatim** here — aarch64 + Jetson's L4T (Linux for Tegra) kernel need a different
dependency story for several packages. This doc is for Track 1 only (CPU inference); GPU
dependencies (Track 3, medium class) are a separate, larger setup and not covered here.

## Baseline

- Flash JetPack (confirms the L4T/CUDA/cuDNN base image and kernel) — record the exact
  JetPack version used once flashed; different JetPack majors ship different default
  Python/OpenCV/CUDA versions, and "which JetPack" needs to be pinned and written down
  here before this doc is trusted as accurate, not assumed.
- Track 1 does not need CUDA/TensorRT at all (CPU-only ONNX inference) — the JetPack
  version mostly matters here for L4T's OpenCV build and general driver/USB stability,
  not for ML acceleration.

## Dependencies — apt vs. pip

| Package | Source | Why not the laptop's route |
|---|---|---|
| OpenCV (`cv2`) | **apt / JetPack-provided**, not `pip install opencv-contrib-python` | The laptop's pip wheel is a generic x86 build; prebuilt aarch64 wheels with the same feature set (V4L2, GTK/Qt for `imshow`) aren't reliably available the same way. JetPack ships a working system OpenCV — use it (verify `cv2.__version__` and `cv2.videoio_registry.getBackends()` include V4L2 before assuming it's usable). |
| `numpy` | pip, no special build needed |
| `onnxruntime` (CPU) | pip — the plain `onnxruntime` package (not `onnxruntime-gpu`) ships aarch64 wheels; confirm the installed version actually has an aarch64 build available for the target Python before assuming parity with the laptop's version. |
| `pyserial` | pip, pure Python, no build concerns |
| Everything else Track 1 imports transitively (`ml_vision`/`ml_audio` reference code) | Match versions already pinned in root `requirements.txt` where a pip wheel exists for aarch64; flag anything that doesn't build cleanly rather than silently swapping in a different version | |

Do **not** `pip install -r requirements.txt` verbatim on the Jetson — several entries there
(`ultralytics`, `torch`, `torchvision`, `openvino`) either need Jetson-specific wheels
(NVIDIA's own Jetson AI Lab / Jetson Zoo pip index for `torch`/`torchvision`) or aren't
needed at all for Track 1 (Track 1 has no PyTorch dependency — it's ONNX + OpenCV +
pyserial only). Building a Track-1-specific `requirements-jetson-track1.txt` once the
above is verified working is a reasonable follow-up, not done as part of this doc.

## Files that must be present on-device

Everything `run_jetson_standalone.py` reads by relative path from `host_software/`, i.e.
this needs the whole `host_software/` tree (or at least `ml_vision/`, `ml_audio/`,
`ml_jetson_vla/`, `src/`, plus repo-root `hardware/platform_templates/`), not just the
`ml_jetson_vla/` directory in isolation:
- `ml_vision/models/shared_vision_backbone_v2/shared_vision_backbone_best.onnx`
- `ml_audio/models/audio_command_classifier_v3.onnx`
- `hardware/platform_templates/ground_truth_manifest.json`

## Verification before trusting this doc

1. `python -c "import cv2; print(cv2.__version__); print(cv2.videoio_registry.getBackends())"`
   — confirm V4L2 is present.
2. `python -c "import onnxruntime as ort; print(ort.get_available_providers())"` — confirm
   at least `CPUExecutionProvider` is present (this is all Track 1 needs).
3. Plug in the USB webcam and STM32, run `runtime/run_jetson_standalone.py --headless`,
   and follow the validation ladder in the Jetson port plan (camera smoke test → ONNX
   parity check → dry run → bench test → on-platform run) before trusting any numbers out
   of it.

This doc has **not yet been executed against real hardware** — it's a bring-up checklist
derived from what Track 1's code actually imports, not a confirmed-working recipe. Update
it with the real JetPack version, actual package versions that worked, and any surprises
once it's been run once on the device.
