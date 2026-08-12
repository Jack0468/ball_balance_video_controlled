"""Package the training code + dataset into one zip for Colab.

Preserves the exact `ml_audio/...` relative directory structure used
locally (e.g. `ml_audio/training/train_audio_command_classifier.py`,
`ml_audio/data/synthetic+real_dataset_large/training_v2/train/...`), so that
after unzipping under a `host_software/` parent directory in Colab, the
training scripts run completely unchanged -- same imports
(`from ml_audio.training.train_audio_command_classifier import ...`), same
`DEFAULT_DATASET_ROOT` resolution via `__file__`, no path overrides needed.

Only the training-relevant files are included -- NOT audio_receiver_pytorch.py
or the live-mic scripts, which pull in sounddevice (needs system PortAudio
libs, irrelevant to training and often annoying to install in a fresh
Colab image). See audio_dsp.py, split out for exactly this reason.

Usage:
    python prepare_colab_package.py
    python prepare_colab_package.py --out-dir <path>
"""

import argparse
import os
import sys
import zipfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ML_AUDIO_DIR = os.path.dirname(SCRIPT_DIR)
HOST_SOFTWARE_DIR = os.path.dirname(ML_AUDIO_DIR)

# Relative to HOST_SOFTWARE_DIR, so the zip's internal paths start with
# "ml_audio/..." -- matching the local package layout exactly.
CODE_FILES = [
    "ml_audio/audio_dsp.py",
    "ml_audio/audio_command_classifier_pytorch.py",
    "ml_audio/evaluations/evaluate_audio_classifier.py",
    "ml_audio/training/audio_augmentations.py",
    "ml_audio/training/train_audio_command_classifier.py",
]
DATASET_DIR = "ml_audio/data/synthetic+real_dataset_large/training_v2"

DEFAULT_OUT_DIR = os.path.join(ML_AUDIO_DIR, "colab_package")
DEFAULT_ZIP_NAME = "ml_audio_colab_package.zip"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zip the training code + training_v2 dataset for upload to Colab."
    )
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--zip-name", default=DEFAULT_ZIP_NAME)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    zip_path = os.path.join(args.out_dir, args.zip_name)

    for rel in CODE_FILES:
        abs_path = os.path.join(HOST_SOFTWARE_DIR, rel)
        if not os.path.isfile(abs_path):
            print(f"ERROR: missing code file {abs_path}", file=sys.stderr)
            sys.exit(1)

    dataset_abs = os.path.join(HOST_SOFTWARE_DIR, DATASET_DIR)
    if not os.path.isdir(dataset_abs):
        print(f"ERROR: missing dataset dir {dataset_abs}", file=sys.stderr)
        sys.exit(1)

    print(f"Writing {zip_path} ...")
    file_count = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in CODE_FILES:
            zf.write(os.path.join(HOST_SOFTWARE_DIR, rel), arcname=rel)
            file_count += 1
            print(f"  + {rel}")

        for root, _dirs, files in os.walk(dataset_abs):
            for fname in files:
                if not fname.lower().endswith(".wav"):
                    continue
                abs_path = os.path.join(root, fname)
                arcname = os.path.relpath(abs_path, HOST_SOFTWARE_DIR).replace(os.sep, "/")
                zf.write(abs_path, arcname=arcname)
                file_count += 1
                if file_count % 2000 == 0:
                    print(f"  ... {file_count} files packed")

    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"Done: {file_count} files, {size_mb:.1f} MB -> {zip_path}")
    print(
        "\nIn Colab:\n"
        "  from google.colab import files\n"
        "  uploaded = files.upload()  # select the zip (or mount Drive and point at it there)\n"
        "  import zipfile, os\n"
        "  os.makedirs('host_software', exist_ok=True)\n"
        f"  zipfile.ZipFile('{args.zip_name}').extractall('host_software')\n"
        "  import sys; sys.path.insert(0, os.path.abspath('host_software'))\n"
    )


if __name__ == "__main__":
    main()
