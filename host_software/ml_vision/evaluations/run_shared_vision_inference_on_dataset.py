"""Offline batch inference of shared_vision_backbone_v2 over a labeled dataset.

Produces the raw-prediction dataset needed to train/evaluate a new MLP time
corrector for this model (see train_mlp_corrector_shared_vision.py). The
existing corrector (mlp_corrector_time_aruco_0730_v1) was trained on the old
model's raw full-camera-frame pixel predictions -- this model's ball_xy output
lives in a different space entirely (normalized [0,1] over the warped 128x128
platform frame), so a fresh prediction dataset is required rather than reusing
Dataset 8's existing ball_x_px/ball_y_px labels (those are ground truth, not
what the model actually predicts -- the corrector needs to learn from the
model's real errors).

Run as a module from the repo root:

    python -m host_software.ml_vision.evaluations.run_shared_vision_inference_on_dataset \
        --csv-file host_software/data/03_gold/shared_vision/labels.csv \
        --images-dir host_software/data/03_gold/shared_vision/images \
        --onnx-model host_software/ml_vision/models/shared_vision_backbone_v2/shared_vision_backbone_best.onnx \
        --output-csv host_software/ml_vision/evaluations/reports/shared_vision_v2_inference_predictions.csv
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd

# Same fixed linear px->mm map as evaluate_shared_vision_backbone.py -- see that
# file's header comment for why this is exact (not an approximation) as long as
# --input-size matches the resolution the images were warped to.
TOUCHPAD_W_MM = 187.5
TOUCHPAD_H_MM = 142.0
PAPER_MARGIN_MM = 6.0


def preprocess(image_bgr: np.ndarray, input_size: tuple) -> np.ndarray:
    """Matches build_eval_transform() + SharedVisionDataset.__getitem__ exactly:
    RGB, resize, [0,1] float, CHW, batch dim. No ImageNet mean/std -- the shared
    vision backbone was never trained with that normalization (unlike the old
    expert_tracker model's preprocess_numpy)."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = input_size
    if image_rgb.shape[:2] != (h, w):
        image_rgb = cv2.resize(image_rgb, (w, h), interpolation=cv2.INTER_AREA)
    image_f = image_rgb.astype(np.float32) / 255.0
    chw = np.transpose(image_f, (2, 0, 1))
    return np.expand_dims(chw, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run shared_vision_backbone_v2 inference over a labeled dataset")
    parser.add_argument("--csv-file", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--onnx-model", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--input-size", type=int, nargs=2, default=[128, 128], help="H W")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N rows (per-session order preserved) -- for dry runs")
    args = parser.parse_args()

    input_size = tuple(args.input_size)
    mm_per_px_x = (TOUCHPAD_W_MM + 2 * PAPER_MARGIN_MM) / input_size[1]
    mm_per_px_y = (TOUCHPAD_H_MM + 2 * PAPER_MARGIN_MM) / input_size[0]

    df = pd.read_csv(args.csv_file, low_memory=False)
    required_cols = {"session", "frame_index", "frame_timestamp_ms", "target_x", "target_y", "touch_x", "touch_y", "image_file"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"--csv-file is missing required columns: {sorted(missing)}")

    # Chronological order within each session -- required for the downstream
    # time-corrector's dt/continuity check to be meaningful.
    df = df.sort_values(["session", "frame_index"]).reset_index(drop=True)
    if args.limit is not None:
        df = df.groupby("session", group_keys=False).head(args.limit).reset_index(drop=True)

    print(f"Loading ONNX model from {args.onnx_model}...")
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 2
    session = ort.InferenceSession(args.onnx_model, sess_options=sess_opts, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    images_dir = Path(args.images_dir)
    rows = []
    t_start = time.perf_counter()
    for i, row in df.iterrows():
        image_path = images_dir / row["image_file"]
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            print(f"[WARN] Could not read {image_path}, skipping.")
            continue

        input_tensor = preprocess(image_bgr, input_size)
        ball_xy = session.run(["ball_xy"], {input_name: input_tensor})[0][0]

        # ball_xy is normalized [0,1] over the warped frame -- first recover the
        # manifest's own mm frame (Y-down, origin at the true paper corner minus
        # margin -- build_paper_corners()'s convention), then convert to the
        # touch_x/touch_y (firmware/PID telemetry) frame. These are NOT related by
        # a simple "subtract half-extent" centering -- confirmed empirically in
        # auto_label_shared_vision.py (the 2026-08-12 "ball label point-reflection
        # bug"): touch_x/touch_y's sign convention is inverted on the X axis
        # relative to the manifest mm frame (ball_x_mm = W/2 - touch_x) but NOT
        # inverted on Y (ball_y_mm = H/2 + touch_y). Inverting that relation:
        #   touch_x = W/2 - manifest_mm_x
        #   touch_y = manifest_mm_y - H/2
        px_x = float(ball_xy[0]) * input_size[1]
        px_y = float(ball_xy[1]) * input_size[0]
        manifest_mm_x = -PAPER_MARGIN_MM + px_x * mm_per_px_x
        manifest_mm_y = -PAPER_MARGIN_MM + px_y * mm_per_px_y
        pred_x = TOUCHPAD_W_MM / 2.0 - manifest_mm_x
        pred_y = manifest_mm_y - TOUCHPAD_H_MM / 2.0

        rows.append(
            {
                "session": row["session"],
                "frame_index": row["frame_index"],
                "frame_timestamp_ms": row["frame_timestamp_ms"],
                "pred_x": pred_x,
                "pred_y": pred_y,
                "target_x": row["target_x"],
                "target_y": row["target_y"],
                "touch_x": row["touch_x"],
                "touch_y": row["touch_y"],
            }
        )

        if (i + 1) % 5000 == 0:
            elapsed = time.perf_counter() - t_start
            print(f"  {i + 1}/{len(df)} frames ({elapsed:.1f}s elapsed, {(i + 1) / elapsed:.1f} fps)")

    out_df = pd.DataFrame(rows)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)

    raw_err = np.sqrt((out_df["pred_x"] - out_df["touch_x"]) ** 2 + (out_df["pred_y"] - out_df["touch_y"]) ** 2)
    print(f"\nWrote {len(out_df)} rows to {output_path}")
    print(f"Raw (uncorrected) mean Euclidean error: {raw_err.mean():.3f} mm (median {raw_err.median():.3f} mm)")


if __name__ == "__main__":
    main()
