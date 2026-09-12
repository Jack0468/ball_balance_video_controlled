# Jetson AGX Orin Environment Setup (Track 1)

Bring-up notes for running `runtime/run_jetson_standalone.py` (small-class expert
pipeline, Phase A) on the Jetson AGX Orin 64GB Developer Kit (p3730). The repo's root
`environment.yml`/`requirements.txt` are x86 (laptop) oriented and **cannot be reused
verbatim** here — aarch64 + Jetson's L4T (Linux for Tegra) kernel need a different
dependency story for several packages. This doc is for Track 1 only (CPU inference); GPU
dependencies (Track 3, medium class) are a separate, larger setup and not covered here.

## Baseline

- **JetPack version: 6.2.3** — chosen 2026-08-19, and **now confirmed flashed onto real
  hardware** (2026-09-08/09), via a fully offline native-Ubuntu-host flash (NVIDIA's raw
  `apply_binaries.sh` → `l4t_flash_prerequisites.sh` → `flash.sh`, not SDK Manager at all —
  see `hardware/jetson_flash/jetson_agx_orin_flash_handoff.md`). Ended up on the same
  version originally planned despite an earlier session's detour toward JetPack 7.x via
  SDK Manager's GUI/CLI — that path is documented as a dead end in
  `JETSON_FLASH_PROCEDURE.md`, not the one actually used. L4T 36.5.2, Ubuntu 22.04, kernel 5.15, CUDA 12.6, TensorRT 10.3,
  cuDNN 9.3 — a minor patch release over 6.2.2 (security/bugfixes), not a different major
  line. Confirmed on NVIDIA's own release notes to support all Jetson Orin modules and dev
  kits, including the AGX Orin. This also unblocks Track 3's GPU path: NVIDIA's Jetson AI
  Lab wheel index for this exact JetPack/CUDA combo
  (`https://pypi.jetson-ai-lab.io/jp6/cu126`) has confirmed prebuilt
  `onnxruntime-gpu==1.23.0`, `torch==2.8.0`, `torchvision==0.23.0` — the JetPack-specific
  wheels Track 3's plan called for, not the generic x86 CUDA wheel.
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
| `pandas` | pip — needed only because `auto_label_shared_vision.py` has a top-level `import pandas`, even though Track 1 never calls the (batch-labeling) function that uses it. **Flagged 2026-09-09 to remove**: that function's pandas usage is 3 trivial lines (`read_csv`/`DataFrame.to_csv`), swap for stdlib `csv` and drop this dependency — not done yet, tracked in `deployment/package_for_jetson.py`'s `REQUIREMENTS` comment. |
| `sounddevice` | pip, **plus `libportaudio2` via apt** — needed by `audio_receiver_onnx.py` for live mic capture; the apt package provides the PortAudio C library `sounddevice` binds to. |
| Everything else Track 1 imports transitively (`ml_vision`/`ml_audio` reference code) | Match versions already pinned in root `requirements.txt` where a pip wheel exists for aarch64; flag anything that doesn't build cleanly rather than silently swapping in a different version | |

Do **not** `pip install -r requirements.txt` verbatim on the Jetson — several entries there
(`ultralytics`, `torch`, `torchvision`, `openvino`) either need Jetson-specific wheels
(NVIDIA's own Jetson AI Lab / Jetson Zoo pip index for `torch`/`torchvision`) or aren't
needed at all for Track 1 (Track 1 has no PyTorch dependency — it's ONNX + OpenCV +
pyserial only). **Built 2026-09-09**: `deployment/package_for_jetson.py` generates a
`requirements-jetson-track1.txt` with exactly the packages above, alongside a minimal file
package (see below) — this superseded the "reasonable follow-up, not done" note that used
to be here.

## Files that must be present on-device

**Don't copy the whole repo — use `deployment/package_for_jetson.py` instead** (built and
verified 2026-09-09, run on the dev machine, not the Jetson):
```bash
python host_software/ml_jetson_vla/deployment/package_for_jetson.py [output_dir]
```
This copies only the ~13-17 files Track 1's import chain actually touches (traced directly,
not guessed — see the script's own docstring for the list and how it was derived) plus the
two ONNX model files, into a directory that mirrors the repo's own top-level layout just
pruned, so every relative-path assumption in the code (`hardware/platform_templates/...`,
`host_software.ml_vision...` absolute imports) resolves with zero code changes. Verified via
`python run_jetson_standalone.py --help` from inside the isolated output directory alone —
confirmed importable standalone, with no dependency on the rest of the repo being present.
Total package size: ~277KB, vs. potentially gigabytes for `host_software/data/` and
everything else in a full repo clone.

**Keep this script's `FILES` list in sync** if `main_onnx_shared_vision_audio.py` (or
anything it imports) gains new dependencies — this has already happened once: the list was
originally traced in August, and by September the file had grown four more imports
(`touch_logger.py`, `keyboard_command_receiver.py`, `scripted_command_sequencer.py`,
`kalman_filter.py`) that the packaging script didn't know about until a real `--help` run
against the packaged copy caught the `ModuleNotFoundError` and revealed them. Re-run that
same verification after any future change to confirm the list is still complete, rather
than trusting it from memory.

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
derived from what Track 1's code actually imports, not a confirmed-working recipe. JetPack
6.2.3 above is a chosen target, not yet confirmed by an actual flash — update this doc with
what the flash/boot actually produced (any version drift, package versions that worked,
surprises) once it's been run once on the device.
