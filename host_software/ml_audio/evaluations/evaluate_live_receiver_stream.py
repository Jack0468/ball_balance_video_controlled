"""Validate the live AudioCommandReceiver end-to-end against a continuous stream.

Unlike evaluate_audio_classifier.py (per-clip, offline), this drives the real
receiver code (rolling buffer, confidence/margin gating, everything) through
data/02_silver/master_evaluation_audio.wav via its source_file playback mode,
so it's testing the actual live decision logic, not a reimplementation of it.

The polling/scoring/report logic lives in live_stream_eval_common.py, shared
with evaluate_nemo_live_receiver_stream.py (the pretrained-backbone track's
equivalent) so both checkpoints are scored by the exact same methodology --
see that module's docstring for the expected command sequence.

Takes ~2 minutes wall-clock since the receiver simulates real-time cadence
on purpose (that's the point of testing it this way).
"""

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ML_AUDIO_DIR = os.path.dirname(SCRIPT_DIR)
HOST_SOFTWARE_DIR = os.path.dirname(ML_AUDIO_DIR)
if HOST_SOFTWARE_DIR not in sys.path:
    sys.path.insert(0, HOST_SOFTWARE_DIR)

from ml_audio.audio_receiver_pytorch import AudioCommandReceiver  # noqa: E402
from ml_audio.evaluations.live_stream_eval_common import run_live_stream_eval  # noqa: E402

DEFAULT_MODEL = os.path.join(
    ML_AUDIO_DIR, "models", "pytorch_v3", "audio_command_classifier_state_dict_v3.pth"
)
DEFAULT_STREAM = os.path.join(ML_AUDIO_DIR, "data", "02_silver", "master_evaluation_audio.wav")
DEFAULT_REPORT_DIR = os.path.join(SCRIPT_DIR, "reports")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Drive the live AudioCommandReceiver through a continuous stream."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--stream", default=DEFAULT_STREAM)
    args = parser.parse_args()

    receiver = AudioCommandReceiver(args.model, source_file=args.stream)
    run_live_stream_eval(
        receiver,
        model_label=os.path.relpath(args.model, ML_AUDIO_DIR),
        stream_label=os.path.relpath(args.stream, ML_AUDIO_DIR),
        out_dir=DEFAULT_REPORT_DIR,
    )


if __name__ == "__main__":
    main()
