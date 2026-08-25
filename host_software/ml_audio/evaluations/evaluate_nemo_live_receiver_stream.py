"""Validate a fine-tuned NeMo checkpoint end-to-end against the same
continuous stream every custom-CNN checkpoint (v3-v7) has been scored
against -- the pretrained-backbone track's equivalent of
evaluate_live_receiver_stream.py.

Runs in Colab (NeMo doesn't install cleanly on Windows -- same reason every
other NeMo step in this plan runs there). Shares the exact scoring/report
logic with the custom-CNN evaluator via live_stream_eval_common.py, so the
two tracks' live-stream numbers are directly comparable, not just
similarly-shaped.

Usage (inside Colab, after extracting the data package and fine-tuning, or
against any .nemo checkpoint copied in):
    python evaluate_nemo_live_receiver_stream.py --model matchboxnet_finetuned.nemo
"""

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ML_AUDIO_DIR = os.path.dirname(SCRIPT_DIR)
HOST_SOFTWARE_DIR = os.path.dirname(ML_AUDIO_DIR)
if HOST_SOFTWARE_DIR not in sys.path:
    sys.path.insert(0, HOST_SOFTWARE_DIR)

from ml_audio.evaluations.live_stream_eval_common import run_live_stream_eval  # noqa: E402
from ml_audio.evaluations.nemo_live_receiver import NemoAudioCommandReceiver  # noqa: E402

DEFAULT_MODEL = os.path.join(ML_AUDIO_DIR, "models", "nemo_matchboxnet_v1", "matchboxnet_finetuned.nemo")
DEFAULT_STREAM = os.path.join(ML_AUDIO_DIR, "data", "02_silver", "master_evaluation_audio.wav")
DEFAULT_REPORT_DIR = os.path.join(SCRIPT_DIR, "reports")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Drive a fine-tuned NeMo checkpoint through the continuous stream test."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--stream", default=DEFAULT_STREAM)
    parser.add_argument("--report-dir", default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()

    receiver = NemoAudioCommandReceiver(args.model, source_file=args.stream)
    run_live_stream_eval(
        receiver,
        model_label=os.path.relpath(args.model, ML_AUDIO_DIR) if args.model.startswith(ML_AUDIO_DIR) else args.model,
        stream_label=os.path.relpath(args.stream, ML_AUDIO_DIR) if args.stream.startswith(ML_AUDIO_DIR) else args.stream,
        out_dir=args.report_dir,
    )


if __name__ == "__main__":
    main()
