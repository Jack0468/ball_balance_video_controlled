"""One-off comparative trial (per ml-vision's experiments/ vs evaluations/
convention -- a comparison report, not a fixed regression benchmark) for
core/color_calibration.py's AdaptiveColorCalibrator, the real fix for
docs/TODO.md's "Marker colour classification needs a robust design, not
another hardcoded fix -- REOPENED 2026-10-14" item.

Reuses the exact real-data verification discipline every prior
marker_classifier.py session established (not a new methodology): real
shared_vision_backbone_v2 ONNX checkpoint -> real MarkerClassifier.classify()
-> nearest-known-feature matching by expected touch_mm position, over real
(not synthetic) photographs from session_20260810_114330 (the one real
session whose 5 features match aruco_markers_03_manifest.json -- there is no
equivalent real multi-frame capture for aruco_markers_01, the sheet actually
mounted live; see the single real logged frame handled separately below).

Three things this script measures, each printed as its own section:

1. REGRESSION CHECK (aruco_markers_03, 1200 real frames): does
   MarkerClassifier(adaptive_calibrator=...) match or beat the existing
   static-HSV-bin baseline once calibrated, on the one large real dataset
   this project has? This is session_20260810_114330, already captured BEFORE
   the sheet/lighting mismatch this fix targets -- static bins were tuned
   against exactly this data, so they are expected to do well here regardless;
   the question is only whether turning calibration on COSTS anything.

2. MIN_BLACK_V_GAP sweep (same dataset): real measured V-gap between
   black_circle and the next-darkest real marker, to pick a MIN_BLACK_V_GAP
   default that is safely below the real gap seen on BOTH the historical
   aruco_markers_03 data and the one real aruco_markers_01 frame (9.4,
   computed from the logged numbers in color_calibration.py's own docstring)
   -- rather than guessing a round number.

3. THE ACTUAL TARGET CASE (aruco_markers_01, 1 real captured frame, replayed
   N times to satisfy the calibrator's warm-up sample count -- NOT N
   independent real observations, stated plainly per this project's own
   "state which evidence class you're using" discipline): does calibration
   resolve the real, documented failure (black_circle -> "red", green_circle
   -> "yellow" under the static bins) where static bins cannot?

Run as a module from the repo root:

    python -m host_software.ml_vision.experiments.marker_classifier_adaptive_calibration_eval
"""

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd

from host_software.ml_vision.core.color_calibration import MIN_BLACK_V_GAP, AdaptiveColorCalibrator
from host_software.ml_vision.core.marker_classifier import MarkerClassifier

REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_03 = REPO_ROOT / "hardware/platform_templates/aruco_markers_03_manifest.json"
LABELS_CSV = REPO_ROOT / "host_software/data/03_gold/shared_vision/labels.csv"
IMAGES_DIR = REPO_ROOT / "host_software/data/03_gold/shared_vision/images"
ONNX_MODEL = REPO_ROOT / "host_software/ml_vision/models/shared_vision_backbone_v2/shared_vision_backbone_best.onnx"
SESSION = "session_20260810_114330"

TOUCHPAD_W_MM = 187.5
TOUCHPAD_H_MM = 142.0
MATCH_RADIUS_MM = 15.0

FEATURE_COLOR = {
    "black_circle": "black",
    "blue_triangle": "blue",
    "yellow_square": "yellow",
    "green_hexagon": "green",
    "red_triangle": "red",
}

# The one real captured frame from the physically-mounted aruco_markers_01
# sheet (docs/PROJECT_LOGBOOK.md 13/10/2026, "Live-hardware verification
# closes..."), not invented: (mean_h, mean_s, mean_v) for each of the 5 real
# features plus the one real spurious 6th blob, and each one's real touch-mm
# position (aruco_markers_01_manifest.json, center=blue/top=black/right=red/
# bottom=green/left=yellow).
REAL_ARUCO_01_FRAME = {
    "black_circle": (7.8, 47.3, 148.2, 0.0, -30.0),
    "yellow_circle": (15.1, 100.9, 234.7, 30.0, 0.0),
    "blue_circle": (129.8, 34.6, 157.9, 0.0, 0.0),
    "red_circle": (1.9, 84.4, 214.8, -30.0, 0.0),
    "green_circle": (34.2, 51.9, 157.6, 0.0, 30.0),
}


def _manifest_to_touch_mm(manifest_mm_x: float, manifest_mm_y: float) -> Tuple[float, float]:
    touch_x = TOUCHPAD_W_MM / 2.0 - manifest_mm_x
    touch_y = manifest_mm_y - TOUCHPAD_H_MM / 2.0
    return touch_x, touch_y


def load_feature_positions() -> Dict[str, Tuple[float, float]]:
    import json

    manifest = json.loads(MANIFEST_03.read_text())
    positions = {}
    for feature in manifest["features"]:
        mx, my = feature["center_mm"]
        positions[feature["name"]] = _manifest_to_touch_mm(mx, my)
    return positions


def preprocess_for_onnx(image_bgr: np.ndarray) -> np.ndarray:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_f = image_rgb.astype(np.float32) / 255.0
    chw = np.transpose(image_f, (2, 0, 1))
    return np.expand_dims(chw, axis=0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def sample_frames(n_frames: Optional[int]) -> pd.DataFrame:
    """Sequential (not evenly-spaced-across-the-whole-session) sample of the
    first `n_frames` real frames -- unlike the green-fallback-sweep harness's
    deterministic-but-scattered sampling, calibration warm-up is a genuinely
    temporal process (it needs to see each marker a few times before it can
    trust anything), so replaying a contiguous prefix is the right simulation
    of what a live session's first few seconds would actually look like."""
    df = pd.read_csv(LABELS_CSV, usecols=["session", "image_file", "frame_index"])
    sub = df[df["session"] == SESSION].sort_values("frame_index").reset_index(drop=True)
    if n_frames is not None and n_frames < len(sub):
        sub = sub.iloc[:n_frames].reset_index(drop=True)
    return sub


def run_harness(
    frames: pd.DataFrame,
    session: ort.InferenceSession,
    input_name: str,
    classifier: MarkerClassifier,
    feature_positions: Dict[str, Tuple[float, float]],
) -> Tuple[Dict[str, Dict[str, int]], Optional[int]]:
    """Returns (per-feature-name counts of what colour each matched blob was
    classified as, the frame index at which the calibrator first reports
    is_calibrated -- None if it never did or no calibrator was supplied)."""
    results: Dict[str, Dict[str, int]] = {name: {} for name in feature_positions}
    calibrated_at: Optional[int] = None

    for frame_idx, (_, row) in enumerate(frames.iterrows()):
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

        if (
            calibrated_at is None
            and classifier.adaptive_calibrator is not None
            and classifier.adaptive_calibrator.is_calibrated
        ):
            calibrated_at = frame_idx

        for det in detections:
            best_name, best_dist = None, float("inf")
            for name, (fx, fy) in feature_positions.items():
                d = math.hypot(det.x_mm - fx, det.y_mm - fy)
                if d < best_dist:
                    best_dist = d
                    best_name = name
            if best_name is not None and best_dist <= MATCH_RADIUS_MM:
                results[best_name][det.color] = results[best_name].get(det.color, 0) + 1

    return results, calibrated_at


def summarize(results: Dict[str, Dict[str, int]], label: str) -> None:
    print(f"  --- {label} ---")
    for feature_name, counts in results.items():
        expected = FEATURE_COLOR[feature_name]
        total = sum(counts.values())
        correct = counts.get(expected, 0)
        pct = 100.0 * correct / total if total else 0.0
        others = {c: n for c, n in sorted(counts.items(), key=lambda kv: -kv[1]) if c != expected}
        print(f"    {feature_name:15s} (expect {expected:6s}): {correct:5d}/{total:5d} ({pct:5.1f}%)  other={others}")


def collect_black_v_gap_stats(
    frames: pd.DataFrame,
    session: ort.InferenceSession,
    input_name: str,
    feature_positions: Dict[str, Tuple[float, float]],
) -> None:
    """Diagnostic: real per-frame mean V for black_circle vs. the next-
    darkest OTHER real marker in that same frame, to pick MIN_BLACK_V_GAP
    against real data instead of a guess. Mirrors
    marker_classifier_green_fallback_sweep.py's collect_hue_stats() shape."""
    plain = MarkerClassifier()
    gaps: List[float] = []

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

        binary_mask = (mask_prob > plain.mask_threshold).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        v_by_feature: Dict[str, float] = {}
        for label_id in range(1, num_labels):
            area = float(stats[label_id, cv2.CC_STAT_AREA])
            if area < plain.min_blob_area_px:
                continue
            blob_mask = (labels == label_id).astype(np.uint8)
            mean_v = float(cv2.mean(hsv_full, mask=blob_mask * 255)[2])
            from host_software.ml_vision.core.marker_classifier import _soft_argmax_centroid

            cx_px, cy_px = _soft_argmax_centroid(heatmap_prob, blob_mask)
            x_mm, y_mm = plain._px_to_touch_mm(cx_px, cy_px)
            best_name, best_dist = None, float("inf")
            for name, (fx, fy) in feature_positions.items():
                d = math.hypot(x_mm - fx, y_mm - fy)
                if d < best_dist:
                    best_dist = d
                    best_name = name
            if best_name is not None and best_dist <= MATCH_RADIUS_MM:
                v_by_feature[best_name] = mean_v

        if "black_circle" in v_by_feature and len(v_by_feature) >= 2:
            black_v = v_by_feature["black_circle"]
            others_v = sorted(v for name, v in v_by_feature.items() if name != "black_circle")
            gaps.append(others_v[0] - black_v)

    arr = np.array(gaps)
    print(f"  real black-vs-next-darkest V gap, n={len(arr)} frames with >=2 matched markers incl. black:")
    if len(arr):
        print(
            f"    mean={arr.mean():.1f} std={arr.std():.1f} min={arr.min():.1f} p5={np.percentile(arr, 5):.1f} "
            f"median={np.median(arr):.1f} p95={np.percentile(arr, 95):.1f} max={arr.max():.1f}"
        )
        print(f"    fraction with gap < {MIN_BLACK_V_GAP:.1f} (shipped MIN_BLACK_V_GAP default): {(arr < MIN_BLACK_V_GAP).mean() * 100:.1f}%")
        print(f"    fraction with gap < 9.4  (the real aruco_markers_01 frame's own gap): {(arr < 9.4).mean() * 100:.1f}%")


def run_real_aruco01_frame_replay(min_black_v_gap: float = MIN_BLACK_V_GAP, n_repeats: int = 10) -> None:
    """THE TARGET CASE. Replays the one real captured aruco_markers_01 frame
    (REAL_ARUCO_01_FRAME, not invented -- see module docstring) `n_repeats`
    times directly through AdaptiveColorCalibrator (not through the ONNX
    model or MarkerClassifier.classify(), since there is only one real image
    file for this frame on disk and this is testing the calibration/labeling
    logic specifically, not blob extraction) -- replaying the SAME real
    observation is not a substitute for genuine multi-frame real data, stated
    plainly; it only satisfies the calibrator's warm-up sample-count gate
    using real (not synthetic) numbers."""
    print(f"  (min_black_v_gap={min_black_v_gap})")
    cal = AdaptiveColorCalibrator(min_samples_per_slot=3, min_slots_for_calibration=2, min_black_v_gap=min_black_v_gap)
    for _ in range(n_repeats):
        for h, s, v, x, y in REAL_ARUCO_01_FRAME.values():
            cal.observe(h, s, v, x, y)
        cal.maybe_calibrate()

    print(f"    calibrated: {cal.is_calibrated}")
    n_correct = 0
    for name, (h, s, v, x, y) in REAL_ARUCO_01_FRAME.items():
        expected = name.split("_")[0]
        got = cal.classify_position(x, y)
        ok = got == expected
        n_correct += int(ok)
        print(f"    {name:15s} (expect {expected:6s}): calibrator says {got}  {'OK' if ok else 'WRONG'}")
    print(f"    {n_correct}/{len(REAL_ARUCO_01_FRAME)} correct")

    # Same frame's STATIC HSV BINS result, for direct before/after comparison
    # (reuses marker_classifier._classify_color() directly on the same real
    # logged numbers -- this is the documented real failure being fixed).
    from host_software.ml_vision.core.marker_classifier import _classify_color

    print("    (static HSV bins on the same real numbers, for comparison):")
    for name, (h, s, v, x, y) in REAL_ARUCO_01_FRAME.items():
        expected = name.split("_")[0]
        got = _classify_color((h, s, v), fallback_hue_threshold=10.0)
        ok = got == expected
        print(f"      {name:15s} (expect {expected:6s}): static bins say {got}  {'OK' if ok else 'WRONG'}")


def main() -> None:
    feature_positions = load_feature_positions()
    ort_session = ort.InferenceSession(str(ONNX_MODEL), providers=["CPUExecutionProvider"])
    input_name = ort_session.get_inputs()[0].name

    n_frames = 1200
    frames = sample_frames(n_frames)
    print(f"Sampled {len(frames)} sequential frames (of 12883) from {SESSION}\n")

    print("=== 1. REGRESSION CHECK (aruco_markers_03, static bins vs. adaptive calibration) ===")
    baseline_classifier = MarkerClassifier()
    baseline_results, _ = run_harness(frames, ort_session, input_name, baseline_classifier, feature_positions)
    summarize(baseline_results, "static HSV bins only (current shipped default)")

    adaptive_classifier = MarkerClassifier(adaptive_calibrator=AdaptiveColorCalibrator())
    adaptive_results, calibrated_at = run_harness(frames, ort_session, input_name, adaptive_classifier, feature_positions)
    summarize(adaptive_results, f"adaptive calibration (calibrated at frame {calibrated_at})")
    print()

    print("=== 2. MIN_BLACK_V_GAP sweep (real aruco_markers_03 data) ===")
    collect_black_v_gap_stats(frames, ort_session, input_name, feature_positions)
    print()

    print("=== 3. THE TARGET CASE: real aruco_markers_01 frame, static bins vs. adaptive calibration ===")
    run_real_aruco01_frame_replay()


if __name__ == "__main__":
    main()
