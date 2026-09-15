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
| OpenCV (`cv2`) | **apt**, not `pip install opencv-contrib-python` | **Confirmed 2026-09-15**: NOT present by default even after the flash — the offline `flash.sh` method only installs the base OS, not the "JetPack SDK Components" (which is what would normally provide `cv2`). Install via `apt-cache search opencv \| grep -i python` to find the right package name for the image (`python3-opencv` on this one), don't assume the name blind. |
| `numpy` | pip, **pinned `<2`** | **Confirmed 2026-09-15**: apt's `python3-opencv` is compiled against numpy's 1.x C ABI. An unpinned `pip install numpy` in 2026 grabs the latest 2.x release, which broke that ABI — surfaced as `AttributeError: _ARRAY_API not found` / `ImportError: numpy.core.multiarray failed to import` on `import cv2`. Fix: `pip3 install "numpy<2"`. Confirmed working combination on real hardware: `cv2==4.5.4`, `numpy==1.26.4`. |
| `onnxruntime` (CPU) — **pin to `1.18.0`, confirmed 2026-09-15** | pip, plain `onnxruntime` package, **not** `onnxruntime-gpu` | Two real, confirmed-on-hardware failures before landing on this: (1) the default/latest `onnxruntime` from PyPI hits a filed upstream bug on Jetson Orin's ARM cores — [microsoft/onnxruntime#28301](https://github.com/microsoft/onnxruntime/issues/28301), `CPUIDInfo` doesn't handle an unrecognized ARM CPU vendor cleanly, surfacing as either a `std::vector` out-of-bounds assertion abort or a `malloc(): invalid size` crash depending on version — described upstream as "100% on Jetson Orin NX, intermittently on AGX Orin," consistent with what we saw. (2) `onnxruntime-gpu` (from either the Jetson AI Lab pip index or NVIDIA's Jetson Zoo) crashes differently (`malloc(): invalid size (unsorted)`) because it needs cuDNN/TensorRT, which the offline flash doesn't install — confirmed via `dpkg -l \| grep cuda` showing only the base `nvidia-l4t-cuda` runtime present, no `/usr/local/cuda`, i.e. no full Toolkit. **Track 1 needs neither GPU package** — plain CPU `onnxruntime==1.18.0` (older than the versions in the filed ARM bug report) loads and runs both models with no crash. Re-verify against a newer version only with real evidence the upstream bug is fixed, not by assumption. |
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
