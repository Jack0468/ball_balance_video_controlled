"""Build a minimal, self-contained copy of just what Track 1
(`runtime/run_jetson_standalone.py`) needs, for transfer to the Jetson --
deliberately excludes `host_software/data/` and everything else in the repo
that Track 1 doesn't actually import.

File list below was derived by tracing `run_jetson_standalone.py`'s real
import chain (including transitive imports of everything it pulls in), not
guessed -- see `host_software/ml_jetson_vla/docs/JETSON_ENV_SETUP.md` for the
matching pip/apt dependency list this package assumes.

The output directory mirrors the repo's own top-level layout (just pruned),
on purpose: every relative-path assumption already in the code
(`hardware/platform_templates/...`, `host_software.ml_vision...` absolute
imports) resolves correctly with zero code changes, because the folder shape
looks identical to the real repo, just smaller.

Usage (run on the dev machine, not the Jetson):
    python package_for_jetson.py [output_dir]
Default output_dir: ./jetson_package_track1/ (sibling to wherever this is run from)
"""

import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]  # deployment/ -> ml_jetson_vla/ -> host_software/ -> repo root

# Relative to repo root. Order doesn't matter; each is copied preserving its
# path under the output directory.
FILES = [
    "host_software/ml_jetson_vla/runtime/run_jetson_standalone.py",
    "host_software/ml_jetson_vla/runtime/session_recorder.py",
    "host_software/ml_jetson_vla/runtime/motor_geometry.py",
    "host_software/ml_jetson_vla/core/policy_interface.py",
    "host_software/ml_jetson_vla/core/control_net.py",
    "host_software/main_onnx_shared_vision_audio.py",
    "host_software/src/receivers.py",
    "host_software/src/utils.py",
    "host_software/src/state_machine.py",
    "host_software/src/audio_receiver_onnx.py",
    "host_software/src/touch_logger.py",
    "host_software/ml_vision/data_processing/auto_label_shared_vision.py",
    "host_software/ml_vision/core/marker_classifier.py",
    "host_software/ml_vision/core/keyboard_command_receiver.py",
    "host_software/ml_vision/core/scripted_command_sequencer.py",
    "host_software/ml_vision/core/kalman_filter.py",
    "hardware/platform_templates/ground_truth_manifest.json",
    "host_software/ml_vision/models/shared_vision_backbone_v2/shared_vision_backbone_best.onnx",
    # This model stores some weight tensors in a separate "external data" file rather
    # than embedding them in the .onnx protobuf -- missing this caused a real, confirmed
    # failure on real hardware (2026-09-15): onnxruntime loaded the .onnx file fine, then
    # failed deserializing a tensor with "Invalid fd was supplied: -1" trying to open this
    # file, which the package never included. `--help`-based import-chain verification
    # doesn't catch this since it never actually loads the ONNX file -- only an actual
    # inference attempt does. If a future model export ever produces a similarly-named
    # `<name>.onnx.data` (or `.onnx_data`) companion file, check for it explicitly rather
    # than assuming a single .onnx file is always self-contained.
    "host_software/ml_vision/models/shared_vision_backbone_v2/shared_vision_backbone_best.onnx.data",
    "host_software/ml_audio/models/audio_command_classifier_v3.onnx",
    # Same external-data situation as the vision model above -- confirmed present in the
    # source repo (2026-09-15) and missing here caused the identical "Invalid fd was
    # supplied: -1" failure, just for conv2_weight in the audio model instead. Any .onnx
    # file in this project should be treated as possibly needing its .onnx.data sibling
    # until proven otherwise -- don't assume single-file self-containment.
    "host_software/ml_audio/models/audio_command_classifier_v3.onnx.data",
    # Not bulk data -- a single 532-byte R/Q calibration result (from
    # estimate_kalman_noise_params.py), needed for --kalman-params to be trustworthy
    # (see kalman_filter.py). Confirmed 2026-09-15 as the best-performing set found so
    # far. This is a deliberate, individually-named exception to "don't copy
    # host_software/data/" -- that exclusion is about the ~85GB bulk tree, not about
    # every file that happens to live under that path.
    "host_software/data/01_bronze/evaluation/kalman_params_20260820_102744.json",
]

# Matches this package's actual import needs -- see the module docstring
# above for how this was derived. `torch` deliberately excluded: only
# imported inside an `if args.mlp:` branch, never at module load time, so
# Track 1's default (--mlp off) path never touches it.
#
# TODO (flagged, not done): `pandas` is only needed because
# auto_label_shared_vision.py has a top-level `import pandas as pd` --
# checked its actual usage (2026-09-09): exactly 3 lines, all trivial
# (`pd.read_csv`, `pd.DataFrame(...).to_csv`), inside a batch dataset-labeling
# function this Jetson path never calls at runtime. The import still executes
# at module load regardless, so pandas has to be installed just to satisfy
# that unused-here function. Worth rewriting those 3 lines to stdlib `csv`
# (not numpy -- the data's mixed string/numeric, csv.DictReader/writer fits
# better) to drop this dependency from the Jetson package entirely -- pandas
# pulls in a much heavier transitive dependency chain than this file's actual
# runtime need justifies. Not done now; this is the note to come back to.
# numpy and onnxruntime are version-pinned deliberately, not just "whatever's newest" --
# both pins came from real crashes on real Jetson AGX Orin hardware (2026-09-15), see
# JETSON_ENV_SETUP.md's dependency table for the full writeup:
#   - numpy<2: apt's python3-opencv is compiled against numpy's 1.x C ABI; an unpinned
#     numpy install grabs 2.x and breaks `import cv2` with "_ARRAY_API not found".
#   - onnxruntime==1.18.0: newer/default onnxruntime hits a filed upstream ARM bug
#     (microsoft/onnxruntime#28301) that crashes on Jetson Orin's CPU vendor detection;
#     onnxruntime-gpu crashes differently since this device has no cuDNN/TensorRT
#     installed (offline-flashed, no JetPack SDK Components). Don't bump either version
#     without re-confirming on real hardware first.
REQUIREMENTS = [
    "numpy<2",
    "onnxruntime==1.18.0",
    "pyserial",
    "sounddevice",
    "pandas",  # see TODO above -- swap for stdlib csv, then remove this line
]


def build_package(output_dir: Path) -> None:
    missing = [f for f in FILES if not (_REPO_ROOT / f).exists()]
    if missing:
        raise FileNotFoundError(
            "Source file(s) not found -- checkout may be incomplete or a path in "
            f"this script's FILES list is stale: {missing}"
        )

    if output_dir.exists():
        print(f"Output directory already exists, removing: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    for rel_path in FILES:
        src = _REPO_ROOT / rel_path
        dst = output_dir / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"  copied: {rel_path}")

    req_path = output_dir / "requirements-jetson-track1.txt"
    req_path.write_text("\n".join(REQUIREMENTS) + "\n")
    print(f"  wrote:  requirements-jetson-track1.txt")

    total_size = sum((output_dir / f).stat().st_size for f in FILES)
    print(f"\nPackage built at {output_dir} -- {len(FILES)} files, {total_size / 1024:.1f} KB total.")
    print(
        "Transfer this whole folder to the Jetson (preserving structure), then on-device:\n"
        "  sudo apt install python3-pip libportaudio2   # system deps -- see JETSON_ENV_SETUP.md for the OpenCV note\n"
        "  pip install -r requirements-jetson-track1.txt\n"
        "  python3 host_software/ml_jetson_vla/runtime/run_jetson_standalone.py --help"
    )


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd() / "jetson_package_track1"
    build_package(out)
