"""Evaluate the shared-vision MLP time corrector: corrected vs. uncorrected
accuracy on the held-out temporal split, so there's a clear answer on whether
mlp_corrector_shared_vision_v1 actually helps before it's wired into
main_onnx_shared_vision_audio.py's --mlp flag.

Adapted from evaluate_mlp_corrector_time.py -- reuses the same held-out
construction as train_mlp_corrector_shared_vision.py (temporal_split, per-
session windowing) so metrics are computed on sequences the corrector never
trained on.

Run as a module from the repo root:

    python -m host_software.ml_vision.evaluations.evaluate_mlp_corrector_shared_vision \
        --csv host_software/ml_vision/evaluations/reports/shared_vision_v2_inference_predictions.csv \
        --model-path host_software/ml_vision/models/mlp_corrector_shared_vision_v1/mlp_corrector_best.pth
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from host_software.ml_vision.training.train_cnn_2d_tracker_marker import temporal_split
from host_software.ml_vision.training.train_mlp_corrector_shared_vision import (
    MLPCorrectorSharedVision,
    TimeWindowDatasetSharedVision,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the shared-vision MLP time corrector")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--future-offset", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--output-dir", default=None, help="Defaults to the checkpoint's parent directory")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {model_path}. Train it first.")
    output_dir = Path(args.output_dir) if args.output_dir else model_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MLPCorrectorSharedVision(window_size=args.window_size)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model = model.to(device)
    model.eval()

    df = pd.read_csv(args.csv)
    _, val_df = temporal_split(df, val_fraction=args.val_fraction, sort_col="frame_index")

    dataset = TimeWindowDatasetSharedVision(val_df, window_size=args.window_size, future_offset=args.future_offset)
    if len(dataset) == 0:
        print("No valid held-out sequences found -- cannot evaluate.")
        return
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    # Uncorrected baseline: the raw pred_x/pred_y at the same target frame each
    # sequence predicts, so this is an apples-to-apples comparison against the
    # corrector's output on identical target frames.
    raw_errors, corrected_errors = [], []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            corrected = model(inputs)
            corrected_errors.append((corrected - targets).cpu().numpy())

            # Reconstruct raw pred_x/pred_y from the last window row's normalized
            # features (index -5:-3 in the flattened [pred_x,pred_y,target_x,target_y,dt]*window vector).
            raw_pred_norm = inputs[:, -5:-3].cpu().numpy()
            raw_pred = raw_pred_norm * np.array([93.75, 71.0])
            raw_errors.append(raw_pred - targets.cpu().numpy())

    raw_errors = np.concatenate(raw_errors, axis=0)
    corrected_errors = np.concatenate(corrected_errors, axis=0)

    raw_euclidean = np.linalg.norm(raw_errors, axis=1)
    corrected_euclidean = np.linalg.norm(corrected_errors, axis=1)

    metrics = {
        "num_val_sequences": int(len(dataset)),
        "raw_mean_euclidean_mm": float(raw_euclidean.mean()),
        "raw_median_euclidean_mm": float(np.median(raw_euclidean)),
        "raw_p95_euclidean_mm": float(np.percentile(raw_euclidean, 95)),
        "corrected_mean_euclidean_mm": float(corrected_euclidean.mean()),
        "corrected_median_euclidean_mm": float(np.median(corrected_euclidean)),
        "corrected_p95_euclidean_mm": float(np.percentile(corrected_euclidean, 95)),
    }
    metrics["improvement_mean_mm"] = metrics["raw_mean_euclidean_mm"] - metrics["corrected_mean_euclidean_mm"]
    metrics["improvement_pct"] = 100.0 * metrics["improvement_mean_mm"] / metrics["raw_mean_euclidean_mm"]

    print("\n--- MLP Corrector (shared_vision_backbone_v2) Evaluation ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.3f}")

    if metrics["improvement_mean_mm"] <= 0:
        print("\n[RESULT] The corrector does NOT improve mean accuracy on this held-out split -- do not enable --mlp by default.")
    else:
        print(f"\n[RESULT] The corrector improves mean accuracy by {metrics['improvement_pct']:.1f}%.")

    metrics_path = output_dir / "evaluation_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"\nSaved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
