"""Build a NeMo-compatible noise manifest from our own background recordings,
for wiring into NeMo's `noise` perturbation (nemo.collections.asr.parts.
preprocessing.perturb.NoisePerturbation) during fine-tuning.

Motivated by the confirmed, 11/11-seeds-across-both-model-sizes `backward`
live-stream failure (see docs/plans/audio_eval_notebook_refactor_plan.md,
"Scaling-Up Result") -- diagnosed via nemo_stream_probe.py as noise-masking
(confidence never rises above `_background_`), not an acoustic confusion or
a capacity problem. The pretrained recipe's inherited augmentor already
mixes in synthetic white noise (`white_noise`, confirmed via
`model.cfg.train_ds.augmentor`) -- this adds a second, `noise` perturbation
sourced from our actual robot/lab recordings instead of synthetic noise, so
fine-tuning sees closer-to-real noise-plus-command mixtures.

NeMo-free (only needs soundfile/scipy, matching nemo_manifest.py's own
"testable locally without touching Colab" design) except this module's one
NeMo-touching piece, `load_mono_16k`, is reused unchanged from
data_processing/segment_background_recording.py rather than reimplemented --
per project convention, extend existing tooling rather than duplicate it.

Confirmed via a local smoke test (nemo_local conda env) before writing this
docstring, not assumed: NoisePerturbation.perturb() loads noise files
directly from disk via AudioSegment.from_file() and raises `ValueError:
Found mismatched channels` if the noise file's channel count doesn't match
the (mono) command clip it's mixing into -- our raw source recordings in
data/01_background_noise/ are stereo, so they cannot be pointed at directly.
This module resolves that once, up front, by writing mono/16kHz cached
copies (reusing the exact downmix/resample already validated for the same
files in segment_background_recording.py) and pointing the manifest at
those instead of the raw stereo originals.
"""

import json
import os
import sys

import soundfile as sf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ML_AUDIO_DIR = os.path.dirname(SCRIPT_DIR)
HOST_SOFTWARE_DIR = os.path.dirname(ML_AUDIO_DIR)
if HOST_SOFTWARE_DIR not in sys.path:
    sys.path.insert(0, HOST_SOFTWARE_DIR)

from ml_audio.audio_dsp import SAMPLE_RATE  # noqa: E402
from ml_audio.data_processing.segment_background_recording import load_mono_16k  # noqa: E402

# The 23-minute lab recording plus the two original robot recordings --
# .m4a excluded, soundfile can't read it directly and load_mono_16k doesn't
# attempt a codec transcode.
NOISE_SOURCE_FILES = [
    os.path.join(ML_AUDIO_DIR, "data", "01_background_noise", "lab_background_sound_01.wav"),
    os.path.join(ML_AUDIO_DIR, "data", "01_background_noise", "robot_background_sound.wav"),
    os.path.join(ML_AUDIO_DIR, "data", "01_background_noise", "robot_background_sound_01.wav"),
]


def build_noise_manifest(cache_dir: str, manifest_path: str) -> list[dict]:
    """Writes mono/16kHz cached copies of NOISE_SOURCE_FILES into cache_dir
    and a NeMo-compatible manifest (one JSON object per line: audio_filepath,
    duration, text) at manifest_path, pointing at those cached copies.
    Returns the list of manifest entries written."""
    os.makedirs(cache_dir, exist_ok=True)
    entries = []
    for source_path in NOISE_SOURCE_FILES:
        stem = os.path.splitext(os.path.basename(source_path))[0]
        cached_path = os.path.abspath(os.path.join(cache_dir, f"{stem}_mono16k.wav"))
        audio = load_mono_16k(source_path)
        sf.write(cached_path, audio, SAMPLE_RATE)
        duration = len(audio) / SAMPLE_RATE
        # NoisePerturbation reads via NeMo's ASRAudioText collection, which
        # expects the standard ASR manifest shape (audio_filepath/duration/
        # text) -- "text" is unused for noise mixing but required for the
        # manifest to parse; a placeholder is fine.
        entries.append({"audio_filepath": cached_path, "duration": duration, "text": "_"})

    with open(manifest_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    return entries


if __name__ == "__main__":
    cache_dir = os.path.join(ML_AUDIO_DIR, "data", "01_background_noise", "_mono16k_cache")
    manifest_path = os.path.join(SCRIPT_DIR, "nemo_noise_manifest.json")
    entries = build_noise_manifest(cache_dir, manifest_path)
    total_hours = sum(e["duration"] for e in entries) / 3600
    print(f"Wrote {len(entries)} noise sources ({total_hours:.2f} hours total) to {manifest_path}")
    for e in entries:
        print(f"  {e['audio_filepath']}  ({e['duration']:.1f}s)")
