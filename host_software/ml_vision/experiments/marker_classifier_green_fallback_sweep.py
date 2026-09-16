"""One-off comparative trial: green S-floor sweep + nearest-hue fallback tier
for marker_classifier.py, following on from docs/PROJECT_LOGBOOK.md's
2026-09-15 "Marker Classifier" entries.

Reuses the exact real-data verification discipline established in that prior
session's work (not a new methodology): real shared_vision_backbone_v2 ONNX
checkpoint -> real MarkerClassifier.classify() -> nearest-known-feature
matching by expected touch_mm position, over real (not synthetic) photographs
from session_20260810_114330 (the one real session whose 5 features match
aruco_markers_03_manifest.json).

This is a comparison report, not a fixed benchmark (per ml-vision's
experiments/ vs evaluations/ convention) -- it exists to justify the S-floor
and fallback-threshold choices actually shipped in marker_classifier.py, not
to be re-run as a regression gate.

Run as a module from the repo root:

    python -m host_software.ml_vision.experiments.marker_classifier_green_fallback_sweep \
        --n-frames 1200
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd

from host_software.ml_vision.core.marker_classifier import (
    COLOR_BINS,
    COLOR_HUE_CENTERS,
    MarkerClassifier,
    _circular_hue_distance,
    _soft_argmax_centroid,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = REPO_ROOT / "hardware/platform_templates/aruco_markers_03_manifest.json"
LABELS_CSV = REPO_ROOT / "host_software/data/03_gold/shared_vision/labels.csv"
IMAGES_DIR = REPO_ROOT / "host_software/data/03_gold/shared_vision/images"
ONNX_MODEL = REPO_ROOT / "host_software/ml_vision/models/shared_vision_backbone_v2/shared_vision_backbone_best.onnx"
SESSION = "session_20260810_114330"

TOUCHPAD_W_MM = 187.5
TOUCHPAD_H_MM = 142.0

# Half the manifest's minimum feature spacing (30mm, quincunx layout) -- used
# to assign each detected blob to the nearest known feature by *position*,
# independent of its color label, exactly as the prior 307-frame harness did.
MATCH_RADIUS_MM = 15.0


def _manifest_to_touch_mm(manifest_mm_x: float, manifest_mm_y: float) -> Tuple[float, float]:
    touch_x = TOUCHPAD_W_MM / 2.0 - manifest_mm_x
    touch_y = manifest_mm_y - TOUCHPAD_H_MM / 2.0
    return touch_x, touch_y


def load_feature_positions() -> Dict[str, Tuple[float, float]]:
    manifest = json.loads(MANIFEST_PATH.read_text())
    positions = {}
    for feature in manifest["features"]:
        mx, my = feature["center_mm"]
        positions[feature["name"]] = _manifest_to_touch_mm(mx, my)
    return positions


FEATURE_COLOR = {
    "black_circle": "black",
    "blue_triangle": "blue",
    "yellow_square": "yellow",
    "green_hexagon": "green",
    "red_triangle": "red",
}


def preprocess_for_onnx(image_bgr: np.ndarray) -> np.ndarray:
    """Matches run_shared_vision_inference_on_dataset.py's preprocess(): RGB,
    [0,1] float, CHW, batch dim. Image is already 128x128 (Dataset 8 gold
    tier), so no resize needed."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_f = image_rgb.astype(np.float32) / 255.0
    chw = np.transpose(image_f, (2, 0, 1))
    return np.expand_dims(chw, axis=0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def sample_frames(n_frames: Optional[int]) -> pd.DataFrame:
    df = pd.read_csv(LABELS_CSV, usecols=["session", "image_file", "frame_index"])
    sub = df[df["session"] == SESSION].sort_values("frame_index").reset_index(drop=True)
    if n_frames is not None and n_frames < len(sub):
        # Evenly spaced, deterministic sample across the whole session (not a
        # random shuffle) -- same rationale as the prior harness's fixed
        # 307-frame sample: reproducible run-to-run so before/after diffs are
        # attributable to the code change, not sampling noise.
        idx = np.linspace(0, len(sub) - 1, n_frames).astype(int)
        sub = sub.iloc[idx].reset_index(drop=True)
    return sub


def run_harness(
    frames: pd.DataFrame,
    session: ort.InferenceSession,
    input_name: str,
    classifier: MarkerClassifier,
    feature_positions: Dict[str, Tuple[float, float]],
) -> Dict[str, Dict[str, int]]:
    """Returns per-feature-name counts of what color each nearby blob was
    classified as, e.g. results["green_hexagon"] = {"green": 812, "unknown": 388, ...}."""
    results: Dict[str, Dict[str, int]] = {name: {} for name in feature_positions}

    for _, row in frames.iterrows():
        image_path = IMAGES_DIR / row["image_file"]
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            continue

        input_tensor = preprocess_for_onnx(image_bgr)
        mask_logits, heatmap_logits = session.run(
            ["mask_logits", "heatmap_logits"], {input_name: input_tensor}
        )
        mask_prob = _sigmoid(mask_logits[0, 0])
        heatmap_prob = _sigmoid(heatmap_logits[0, 0])

        detections = classifier.classify(image_bgr, mask_prob, heatmap_prob)

        for det in detections:
            best_name, best_dist = None, float("inf")
            for name, (fx, fy) in feature_positions.items():
                d = math.hypot(det.x_mm - fx, det.y_mm - fy)
                if d < best_dist:
                    best_dist = d
                    best_name = name
            if best_name is not None and best_dist <= MATCH_RADIUS_MM:
                results[best_name][det.color] = results[best_name].get(det.color, 0) + 1

    return results


def collect_hue_stats(
    frames: pd.DataFrame,
    session: ort.InferenceSession,
    input_name: str,
    feature_positions: Dict[str, Tuple[float, float]],
    mask_threshold: float = 0.5,
    min_blob_area_px: float = 4.0,
) -> Dict[str, List[float]]:
    """Diagnostic-only: for every blob matched to a known feature location
    (regardless of what color it currently classifies as, or whether it
    classifies at all), record its raw mean hue. Duplicates a few lines of
    marker_classifier.classify()'s blob-extraction logic rather than calling
    it, since MarkerDetection doesn't carry hue and this is throwaway
    diagnostic code, not something to grow the shipped dataclass for.
    Purpose: characterize whether black_circle's real hue is a stable value
    (like the other 4 colors) or unconstrained scatter -- decides whether
    "black" belongs in the nearest-hue fallback tier at all."""
    hue_by_feature: Dict[str, List[float]] = {name: [] for name in feature_positions}
    classifier = MarkerClassifier()  # only used for its _px_to_touch_mm helper

    for _, row in frames.iterrows():
        image_path = IMAGES_DIR / row["image_file"]
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            continue

        input_tensor = preprocess_for_onnx(image_bgr)
        mask_logits, heatmap_logits = session.run(
            ["mask_logits", "heatmap_logits"], {input_name: input_tensor}
        )
        mask_prob = _sigmoid(mask_logits[0, 0])
        heatmap_prob = _sigmoid(heatmap_logits[0, 0])
        hsv_full = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

        binary_mask = (mask_prob > mask_threshold).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        for label in range(1, num_labels):
            area = float(stats[label, cv2.CC_STAT_AREA])
            if area < min_blob_area_px:
                continue
            blob_mask = (labels == label).astype(np.uint8)
            mean_hsv = cv2.mean(hsv_full, mask=blob_mask * 255)[:3]
            cx_px, cy_px = _soft_argmax_centroid(heatmap_prob, blob_mask)
            x_mm, y_mm = classifier._px_to_touch_mm(cx_px, cy_px)

            best_name, best_dist = None, float("inf")
            for name, (fx, fy) in feature_positions.items():
                d = math.hypot(x_mm - fx, y_mm - fy)
                if d < best_dist:
                    best_dist = d
                    best_name = name
            if best_name is not None and best_dist <= MATCH_RADIUS_MM:
                hue_by_feature[best_name].append(float(mean_hsv[0]))

    return hue_by_feature


def summarize_hue_stats(hue_by_feature: Dict[str, List[float]]) -> None:
    for name, hues in hue_by_feature.items():
        if not hues:
            print(f"  {name:15s}: no matched blobs")
            continue
        arr = np.array(hues)
        expected_color = FEATURE_COLOR[name]
        center = COLOR_HUE_CENTERS.get(expected_color)
        if center is not None:
            dists = np.array([_circular_hue_distance(h, center) for h in hues])
            spread_desc = f"circ-dist-to-own-center: mean={dists.mean():.1f} std={dists.std():.1f} p95={np.percentile(dists, 95):.1f}"
        else:
            spread_desc = "(no canonical center -- achromatic)"
        print(
            f"  {name:15s} n={len(hues):4d}  raw-hue: mean={arr.mean():5.1f} std={arr.std():5.1f} "
            f"min={arr.min():5.1f} p5={np.percentile(arr, 5):5.1f} median={np.median(arr):5.1f} "
            f"p95={np.percentile(arr, 95):5.1f} max={arr.max():5.1f}  {spread_desc}"
        )


def sweep_fallback_threshold(
    frames: pd.DataFrame,
    session: ort.InferenceSession,
    input_name: str,
    feature_positions: Dict[str, Tuple[float, float]],
    thresholds: List[float],
) -> None:
    for threshold in thresholds:
        print(f"\n=== FALLBACK HUE THRESHOLD = {threshold} ===")
        classifier = MarkerClassifier(fallback_hue_threshold=threshold)
        results = run_harness(frames, session, input_name, classifier, feature_positions)
        summarize(results)


def summarize(results: Dict[str, Dict[str, int]]) -> None:
    for feature_name, counts in results.items():
        expected = FEATURE_COLOR[feature_name]
        total = sum(counts.values())
        correct = counts.get(expected, 0)
        pct = 100.0 * correct / total if total else 0.0
        others = {c: n for c, n in sorted(counts.items(), key=lambda kv: -kv[1]) if c != expected}
        print(f"  {feature_name:15s} (expect {expected:6s}): {correct:5d}/{total:5d} ({pct:5.1f}%)  other={others}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-frames", type=int, default=1200)
    parser.add_argument("--sweep-s-floor", action="store_true", default=False)
    parser.add_argument("--hue-stats", action="store_true", default=False)
    parser.add_argument("--sweep-fallback", action="store_true", default=False)
    args = parser.parse_args()

    feature_positions = load_feature_positions()
    frames = sample_frames(args.n_frames)
    print(f"Sampled {len(frames)}/12883 frames from {SESSION}")

    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 2
    ort_session = ort.InferenceSession(str(ONNX_MODEL), sess_options=sess_opts, providers=["CPUExecutionProvider"])
    input_name = ort_session.get_inputs()[0].name

    # --- Baseline: current shipped bins, no fallback (fallback_hue_threshold=0) ---
    print("\n=== BASELINE (current shipped COLOR_BINS, fallback disabled) ===")
    baseline_classifier = MarkerClassifier(fallback_hue_threshold=0.0)
    baseline_results = run_harness(frames, ort_session, input_name, baseline_classifier, feature_positions)
    summarize(baseline_results)

    # --- Green S-floor sweep (fallback still disabled, isolate this change) ---
    # NOTE: COLOR_BINS["green"] currently reflects whatever is hardcoded in
    # marker_classifier.py at import time (S floor 20, shipped after this
    # sweep's own first run). Re-sweeping here re-confirms the shipped choice
    # rather than assuming it.
    original_green_bin = COLOR_BINS["green"]
    if args.sweep_s_floor:
        for s_floor in (15, 20, 25, 30, 35, 40, 45, 50):
            COLOR_BINS["green"] = [(np.array([45, s_floor, 50]), np.array([85, 255, 255]))]
            print(f"\n=== GREEN S-FLOOR = {s_floor} (fallback disabled) ===")
            classifier = MarkerClassifier(fallback_hue_threshold=0.0)
            results = run_harness(frames, ort_session, input_name, classifier, feature_positions)
            summarize(results)
        COLOR_BINS["green"] = original_green_bin

    # --- Diagnostic: is black's real hue a stable value or unconstrained
    # scatter? Decides whether black belongs in the fallback tier at all. ---
    if args.hue_stats:
        print("\n=== RAW HUE STATS PER FEATURE (diagnostic, informs fallback-tier design) ===")
        hue_stats = collect_hue_stats(frames, ort_session, input_name, feature_positions)
        summarize_hue_stats(hue_stats)

    # --- Fallback-hue-threshold sweep, on top of the shipped S=20 green bin.
    # threshold=0.0 reproduces the strict-bins-only baseline (no fallback). ---
    if args.sweep_fallback:
        sweep_fallback_threshold(
            frames, ort_session, input_name, feature_positions,
            thresholds=[0.0, 5.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0],
        )


if __name__ == "__main__":
    main()
