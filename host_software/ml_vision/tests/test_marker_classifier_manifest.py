"""
test_marker_classifier_manifest.py

Regression tests for the 13/10/2026 sheet-identity fix to
`host_software/ml_vision/core/marker_classifier.py` (`MarkerClassifier`'s new
opt-in `manifest_path` param + `_load_feature_positions_touch_mm()`).

Background (see docs/PROJECT_LOGBOOK.md 13/10/2026, "Live-hardware
verification closes: black/green still never appear post-fix..."): the
classifier's HSV bins have now twice been re-tuned against one specific
historical dataset (`session_20260810_114330`, sheet `aruco_markers_03`) and
broken again on a different physical sheet/lighting (`aruco_markers_01`,
mounted on the Jetson) -- judged "not fixable by bin tuning" in that entry,
because the failure is a confounded sheet-identity + global colour-cast shift,
not an isolated per-colour miscalibration. `MarkerClassifier` already computes
each blob's (x_mm, y_mm) via the ArUco-homography-derived soft-argmax centroid
-- that same investigation confirmed these positions matched the manifest to
<0.5mm even on the frame where every HSV bin failed. The fix: when a sheet
manifest is supplied, identify a blob by its already-known, lighting-invariant
physical slot position first, and only fall back to the original HSV bins
when no manifest is loaded or no slot matches closely enough.

The two "real measured HSV" cases below are not invented -- they are the
exact (mean_h, mean_s, mean_v) values logged in that entry's results table for
`aruco_markers_01`'s `black_circle` (misclassified "red") and `green_circle`
(misclassified "yellow") blobs, so these tests reproduce the real documented
failure first, then confirm the manifest-position fix resolves it.
"""

import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from core.marker_classifier import (  # noqa: E402
    PAPER_MARGIN_MM,
    TOUCHPAD_H_MM,
    TOUCHPAD_W_MM,
    MarkerClassifier,
    _load_feature_positions_touch_mm,
)

_MANIFEST_01 = (
    Path(__file__).resolve().parents[3]
    / "hardware"
    / "platform_templates"
    / "aruco_markers_01_manifest.json"
)

_INPUT_SIZE = (128, 128)
_BACKGROUND_HSV = (0, 0, 230)  # blank paper: desaturated, bright -- never classifies as any color.
_BLOB_RADIUS_PX = 6


def _hsv_to_bgr(h: float, s: float, v: float) -> Tuple[int, int, int]:
    pixel = np.uint8([[[round(h), round(s), round(v)]]])
    bgr = cv2.cvtColor(pixel, cv2.COLOR_HSV2BGR)
    return int(bgr[0, 0, 0]), int(bgr[0, 0, 1]), int(bgr[0, 0, 2])


def _touch_mm_to_px(classifier: MarkerClassifier, x_mm: float, y_mm: float) -> Tuple[int, int]:
    """Inverse of MarkerClassifier._px_to_touch_mm(), so a test can place a
    synthetic blob at a known touch-mm position (e.g. a manifest feature's
    own position) instead of an arbitrary pixel coordinate."""
    manifest_mm_x = TOUCHPAD_W_MM / 2.0 - x_mm
    manifest_mm_y = y_mm + TOUCHPAD_H_MM / 2.0
    px_x = (manifest_mm_x + PAPER_MARGIN_MM) / classifier._mm_per_px_x
    px_y = (manifest_mm_y + PAPER_MARGIN_MM) / classifier._mm_per_px_y
    return int(round(px_x)), int(round(px_y))


def _build_frame_with_blobs(
    classifier: MarkerClassifier, blobs_touch_mm_and_hsv: list
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Builds (warped_frame_bgr, mask_prob, heatmap_prob) with one filled
    circular blob per (x_mm, y_mm, (h, s, v)) entry, on a blank background."""
    h, w = _INPUT_SIZE
    hsv_frame = np.zeros((h, w, 3), dtype=np.uint8)
    hsv_frame[:, :, 0] = _BACKGROUND_HSV[0]
    hsv_frame[:, :, 1] = _BACKGROUND_HSV[1]
    hsv_frame[:, :, 2] = _BACKGROUND_HSV[2]
    mask_prob = np.zeros((h, w), dtype=np.float32)

    for x_mm, y_mm, (bh, bs, bv) in blobs_touch_mm_and_hsv:
        px_x, px_y = _touch_mm_to_px(classifier, x_mm, y_mm)
        cv2.circle(hsv_frame, (px_x, px_y), _BLOB_RADIUS_PX, (int(bh), int(bs), int(bv)), -1)
        cv2.circle(mask_prob, (px_x, px_y), _BLOB_RADIUS_PX, 1.0, -1)

    warped_frame_bgr = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2BGR)
    heatmap_prob = mask_prob.copy()  # uniform weight inside each blob -> soft-argmax == plain centroid
    return warped_frame_bgr, mask_prob, heatmap_prob


def test_load_feature_positions_matches_known_aruco_markers_01_layout() -> None:
    """Sanity check: the loader's touch-mm conversion must match the manifest
    layout confirmed live on the Jetson (docs/PROJECT_LOGBOOK.md 13/10/2026):
    center=blue, top=black, right=red, bottom=green, left=yellow."""
    features = _load_feature_positions_touch_mm(_MANIFEST_01)
    by_color = {f["color"]: (f["x_mm"], f["y_mm"]) for f in features}

    assert by_color["blue"] == (0.0, 0.0)
    assert by_color["black"] == (0.0, -30.0)
    assert by_color["red"] == (-30.0, 0.0)
    assert by_color["green"] == (0.0, 30.0)
    assert by_color["yellow"] == (30.0, 0.0)


def test_manifest_path_defaults_to_none_and_preserves_hsv_only_behavior() -> None:
    """No manifest supplied (every existing call site's behavior, unchanged)
    -> classify() must fall straight through to the original HSV bins, same
    as before this fix existed."""
    classifier = MarkerClassifier(input_size=_INPUT_SIZE)
    assert classifier._manifest_features == []

    # A real, correctly-saturated red reading (well inside the strict red bin).
    warped_bgr, mask_prob, heatmap_prob = _build_frame_with_blobs(
        classifier, [(-30.0, 0.0, (2.0, 200.0, 220.0))]
    )
    detections = classifier.classify(warped_bgr, mask_prob, heatmap_prob)

    assert len(detections) == 1
    assert detections[0].color == "red"
    assert detections[0].classification_source == "hsv_bins"


def test_real_logged_hsv_failure_is_reproduced_without_manifest() -> None:
    """Reproduces the exact real failure from docs/PROJECT_LOGBOOK.md's
    13/10/2026 measured table (not invented values): black_circle's own blob
    (mean_h=7.8, mean_s=47.3, mean_v=148.2) reads as "red", and green_circle's
    own blob (mean_h=34.2, mean_s=51.9, mean_v=157.6) reads as "yellow" -- both
    via the strict HSV bins, exactly as logged."""
    classifier = MarkerClassifier(input_size=_INPUT_SIZE)

    warped_bgr, mask_prob, heatmap_prob = _build_frame_with_blobs(
        classifier,
        [
            (0.0, -30.0, (7.8, 47.3, 148.2)),   # true identity: black_circle (top slot)
            (0.0, 30.0, (34.2, 51.9, 157.6)),    # true identity: green_circle (bottom slot)
        ],
    )
    detections = classifier.classify(warped_bgr, mask_prob, heatmap_prob)
    colors_by_slot = {(round(d.y_mm)): d.color for d in detections}

    assert colors_by_slot[-30] == "red"      # WRONG -- the real, logged failure
    assert colors_by_slot[30] == "yellow"    # WRONG -- the real, logged failure


def test_manifest_position_fixes_the_real_logged_hsv_failure() -> None:
    """Same exact synthetic frame as the previous test (same real logged HSV
    values), but with aruco_markers_01's manifest supplied -- both blobs must
    now resolve to their TRUE color via position, not the wrong HSV-bin color,
    with classification_source correctly reporting the manifest tier."""
    classifier = MarkerClassifier(input_size=_INPUT_SIZE, manifest_path=_MANIFEST_01)
    assert len(classifier._manifest_features) == 5

    warped_bgr, mask_prob, heatmap_prob = _build_frame_with_blobs(
        classifier,
        [
            (0.0, -30.0, (7.8, 47.3, 148.2)),   # true identity: black_circle (top slot)
            (0.0, 30.0, (34.2, 51.9, 157.6)),    # true identity: green_circle (bottom slot)
        ],
    )
    detections = classifier.classify(warped_bgr, mask_prob, heatmap_prob)
    by_slot = {round(d.y_mm): d for d in detections}

    assert by_slot[-30].color == "black"
    assert by_slot[-30].classification_source == "manifest_position"
    assert by_slot[30].color == "green"
    assert by_slot[30].classification_source == "manifest_position"


def test_manifest_match_radius_rejects_far_off_blob() -> None:
    """A blob nowhere near any manifest slot (e.g. the wrong sheet is loaded,
    or a spurious CNN mask blob) must NOT be claimed by the manifest tier --
    it should fall through to the HSV bins exactly like the no-manifest case,
    not be forced onto the nearest slot regardless of distance."""
    classifier = MarkerClassifier(input_size=_INPUT_SIZE, manifest_path=_MANIFEST_01)

    # 50mm from every real aruco_markers_01 slot -- outside manifest_match_radius_mm (15.0).
    warped_bgr, mask_prob, heatmap_prob = _build_frame_with_blobs(
        classifier, [(50.0, 50.0, (2.0, 200.0, 220.0))]
    )
    detections = classifier.classify(warped_bgr, mask_prob, heatmap_prob)

    assert len(detections) == 1
    assert detections[0].classification_source == "hsv_bins"
    assert detections[0].color == "red"  # same strong red reading as the no-manifest test


def test_manifest_claim_is_unique_and_prefers_larger_blob() -> None:
    """Two blobs competing for the same manifest slot (e.g. a real marker plus
    a small spurious blob landing nearby) must not both claim it -- the larger
    (real) blob wins the slot; the smaller one falls back to its own HSV
    reading rather than silently duplicating the slot's color."""
    classifier = MarkerClassifier(input_size=_INPUT_SIZE, manifest_path=_MANIFEST_01)

    h, w = _INPUT_SIZE
    hsv_frame = np.zeros((h, w, 3), dtype=np.uint8)
    hsv_frame[:, :, 0] = _BACKGROUND_HSV[0]
    hsv_frame[:, :, 1] = _BACKGROUND_HSV[1]
    hsv_frame[:, :, 2] = _BACKGROUND_HSV[2]
    mask_prob = np.zeros((h, w), dtype=np.float32)

    # Real blue_circle at the true center slot (0, 0) -- larger, radius 3
    # (extends +/-3px from center).
    px_x, px_y = _touch_mm_to_px(classifier, 0.0, 0.0)
    cv2.circle(hsv_frame, (px_x, px_y), 3, (120, 200, 220), -1)
    cv2.circle(mask_prob, (px_x, px_y), 3, 1.0, -1)

    # A smaller spurious blob 7px away (~10.9mm -- still within blue's own
    # 15mm manifest_match_radius_mm, and far enough from red's slot at -30mm
    # not to accidentally match THAT one instead), with a 1px background gap
    # from the real blob so connectedComponentsWithStats sees two components,
    # not one merged blob. Painted a strong, unambiguous red.
    spurious_px = (px_x + 7, px_y)
    cv2.circle(hsv_frame, spurious_px, 2, (2, 200, 220), -1)
    cv2.circle(mask_prob, spurious_px, 2, 1.0, -1)

    warped_bgr = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2BGR)
    heatmap_prob = mask_prob.copy()

    detections = classifier.classify(warped_bgr, mask_prob, heatmap_prob)
    assert len(detections) == 2

    by_area = sorted(detections, key=lambda d: d.area_px, reverse=True)
    larger, smaller = by_area[0], by_area[1]

    assert larger.classification_source == "manifest_position"
    assert larger.color == "blue"
    # The smaller blob could not claim the already-claimed center slot, so it
    # must fall back to its own (correct, strongly-red) HSV reading instead of
    # also being labeled "blue".
    assert smaller.classification_source == "hsv_bins"
    assert smaller.color == "red"
