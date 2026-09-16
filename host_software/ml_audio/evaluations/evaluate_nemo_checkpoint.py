"""Offline confusion-matrix eval + .nemo/.onnx export for a NeMo checkpoint,
runnable locally (NeMo installs cleanly on Windows -- see plan doc) against
either a raw Lightning `.ckpt` (what ModelCheckpoint writes mid-training,
before the notebook's own export cell ever runs) or an already-exported
`.nemo` file.

Built because a Colab session can disconnect after ModelCheckpoint has saved
a checkpoint but before the notebook reaches its offline-eval/export cells
(exactly what happened to the first two NOISE_MIX runs, seed0/seed1 --
`models/checkpoints_3x1x64_noisemix_seed{0,1}/*.ckpt` with no matching
confusion-matrix report or .nemo/.onnx export anywhere) -- rather than
re-running the whole notebook, this reuses the identical eval/export logic
those cells already contain (same per-file confusion-matrix loop, same
JSON report schema, same .save_to()/.export() calls) as a standalone local
script, so a `.ckpt`-only checkpoint doesn't need a full Colab round-trip
just to find out if it's worth keeping.

Usage:
    python evaluate_nemo_checkpoint.py --checkpoint models/checkpoints_3x1x64_noisemix_seed0/matchboxnet_3x1x64_noisemix_finetuned-seed0-epoch=016-val_acc_micro_top_1=0.9575.ckpt --run-tag 3x1x64_noisemix --seed 0
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import soundfile as sf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ML_AUDIO_DIR = os.path.dirname(SCRIPT_DIR)
HOST_SOFTWARE_DIR = os.path.dirname(ML_AUDIO_DIR)
if HOST_SOFTWARE_DIR not in sys.path:
    sys.path.insert(0, HOST_SOFTWARE_DIR)

from ml_audio.training.nemo_manifest import build_manifest_entries  # noqa: E402
from ml_audio.training.train_audio_command_classifier import DEFAULT_DATASET_ROOT, discover_labels  # noqa: E402

DEFAULT_REPORT_DIR = os.path.join(SCRIPT_DIR, "reports")
DEFAULT_MODELS_DIR = os.path.join(ML_AUDIO_DIR, "models")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline-eval and export a NeMo checkpoint (.ckpt or .nemo)."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to a .ckpt or .nemo file.")
    parser.add_argument("--run-tag", required=True, help="e.g. 3x1x64_noisemix -- used in output filenames/dirs.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--report-dir", default=DEFAULT_REPORT_DIR)
    parser.add_argument("--models-dir", default=DEFAULT_MODELS_DIR)
    parser.add_argument("--no-export", action="store_true", help="Skip writing .nemo/.onnx, just report the confusion matrix.")
    args = parser.parse_args()

    import torch
    import nemo.collections.asr as nemo_asr

    print(f"Loading {args.checkpoint} ...")
    if args.checkpoint.endswith(".ckpt"):
        model = nemo_asr.models.EncDecClassificationModel.load_from_checkpoint(args.checkpoint)
    else:
        model = nemo_asr.models.EncDecClassificationModel.restore_from(args.checkpoint)
    model.eval()
    device = next(model.parameters()).device
    labels = list(model.cfg.labels)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded on {device}. {n_params:,} params. Labels ({len(labels)}): {labels}")

    # Confirm this checkpoint's labels actually match the dataset's own
    # alphabetical discovery -- same assertion discipline as the notebook,
    # not assumed just because it loaded without error.
    dataset_labels = discover_labels(args.dataset_root)
    assert labels == dataset_labels, (
        f"checkpoint labels {labels} don't match dataset labels {dataset_labels} -- "
        "wrong checkpoint or wrong dataset root?"
    )

    val_entries = build_manifest_entries(args.dataset_root, "val", labels)
    print(f"Evaluating against {len(val_entries)} val clips...")

    label_to_idx = {l: i for i, l in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    with torch.no_grad():
        for entry in val_entries:
            audio, sr = sf.read(entry["audio_filepath"], dtype="float32")
            assert sr == 16000
            audio_t = torch.as_tensor(audio, device=device).unsqueeze(0)
            len_t = torch.tensor([len(audio)], device=device)
            logits = model(input_signal=audio_t, input_signal_length=len_t)
            pred_idx = int(logits.argmax(dim=-1).item())
            true_idx = label_to_idx[entry["label"]]
            matrix[true_idx, pred_idx] += 1

    correct = int(np.trace(matrix))
    total = len(val_entries)
    print(f"\nOffline accuracy: {correct/total:.3%} ({correct}/{total})")
    print(f"\n{'label':14s} recall")
    for i, label in enumerate(labels):
        row_total = matrix[i].sum()
        recall = matrix[i, i] / row_total if row_total else 0.0
        print(f"{label:14s} {matrix[i, i]}/{row_total} = {recall:.1%}")

    os.makedirs(args.report_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = os.path.join(args.report_dir, f"nemo_finetune_confusion_matrix_{args.run_tag}_seed{args.seed}_{timestamp}.json")
    with open(report_path, "w") as f:
        json.dump({
            "generated_at": timestamp,
            "model_variant": args.run_tag,
            "n_params": n_params,
            "seed": args.seed,
            "checkpoint": os.path.abspath(args.checkpoint),
            "labels": labels,
            "accuracy": correct / total,
            "total_clips": total,
            "correct": correct,
            "matrix": matrix.tolist(),
        }, f, indent=2)
    print(f"\nSaved: {report_path}")

    if args.no_export:
        return

    out_dir = os.path.join(args.models_dir, f"nemo_matchboxnet_{args.run_tag}_seed{args.seed}")
    os.makedirs(out_dir, exist_ok=True)
    nemo_path = os.path.join(out_dir, f"matchboxnet_{args.run_tag}_finetuned_seed{args.seed}.nemo")
    onnx_path = os.path.join(out_dir, f"matchboxnet_{args.run_tag}_finetuned_seed{args.seed}.onnx")
    model.save_to(nemo_path)
    model.export(onnx_path)
    print(f"Exported: {nemo_path}\nExported: {onnx_path}")


if __name__ == "__main__":
    main()
