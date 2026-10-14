"""
test_marker_classifier_adaptive_calibration.py

Integration tests for MarkerClassifier's opt-in `adaptive_calibrator`
parameter (2026-10-14 REOPENED fix for docs/TODO.md's "Marker colour
classification needs a robust design" item) -- the real live-operation
replacement for the reverted `manifest_path` ground-truth lookup. See
core/color_calibration.py's module docstring for the full method, and
test_color_calibration.py for unit tests of the calibration algorithm
itself; this file tests MarkerClassifier.classify()'s wiring of it.

Reuses `test_marker_classifier_manifest.py`'s exact synthetic-frame-building
helpers and its real-logged-HSV-value convention (the two cases reproduced
below -- black_circle misclassified "red", green_circle misclassified
"yellow" -- are the same real numbers that file already uses, not invented).
"""

import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from core.color_calibration import AdaptiveColorCalibrator  # noqa: E402
from core.marker_classifier import (  # noqa: E402
    PAPER_MARGIN_MM,
    TOUCHPAD_H_MM,
    TOUCHPAD_W_MM,
    MarkerClassifier,
)

_INPUT_SIZE = (128, 128)
_BACKGROUND_HSV = (0, 0, 230)
_BLOB_RADIUS_PX = 6

# Real logged aruco_markers_01 frame (docs/PROJECT_LOGBOOK.md 13/10/2026) --
# same convention/positions as test_marker_classifier_manifest.py.
REAL_ARUCO_01_FRAME = {
    "black": (0.0, -30.0, (7.8, 47.3, 148.2)),
    "yellow": (30.0, 0.0, (15.1, 100.9, 234.7)),
    "blue": (0.0, 0.0, (129.8, 34.6, 157.9)),
    "red": (-30.0, 0.0, (1.9, 84.4, 214.8)),
    "green": (0.0, 30.0, (34.2, 51.9, 157.6)),
}


def _touch_mm_to_px(classifier: MarkerClassifier, x_mm: float, y_mm: float) -> Tuple[int, int]:
    manifest_mm_x = TOUCHPAD_W_MM / 2.0 - x_mm
    manifest_mm_y = y_mm + TOUCHPAD_H_MM / 2.0
    px_x = (manifest_mm_x + PAPER_MARGIN_MM) / classifier._mm_per_px_x
    px_y = (manifest_mm_y + PAPER_MARGIN_MM) / classifier._mm_per_px_y
    return int(round(px_x)), int(round(px_y))


def _build_frame_with_blobs(
    classifier: MarkerClassifier, blobs_touch_mm_and_hsv: list
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    heatmap_prob = mask_prob.copy()
    return warped_frame_bgr, mask_prob, heatmap_prob


def _real_frame_blob_list() -> list:
    return [(x, y, hsv) for (x, y, hsv) in REAL_ARUCO_01_FRAME.values()]


def test_adaptive_calibrator_none_by_default_preserves_existing_behavior() -> None:
    """Every existing call site constructs MarkerClassifier() without this
    argument -- confirms it's still None and classify() is untouched."""
    classifier = MarkerClassifier(input_size=_INPUT_SIZE)
    assert classifier.adaptive_calibrator is None

    warped_bgr, mask_prob, heatmap_prob = _build_frame_with_blobs(
        classifier, [(-30.0, 0.0, (2.0, 200.0, 220.0))]
    )
    detections = classifier.classify(warped_bgr, mask_prob, heatmap_prob)
    assert len(detections) == 1
    assert detections[0].color == "red"
    assert detections[0].classification_source == "hsv_bins"


def test_calibrator_not_yet_warmed_up_falls_through_to_hsv_bins() -> None:
    """A single frame is not enough to satisfy the calibrator's warm-up gate
    (min_samples_per_slot=5 default) -- classify() must still return a usable
    (if possibly wrong, same as today) HSV-bin answer, not crash or stall."""
    calibrator = AdaptiveColorCalibrator()
    classifier = MarkerClassifier(input_size=_INPUT_SIZE, adaptive_calibrator=calibrator)

    warped_bgr, mask_prob, heatmap_prob = _build_frame_with_blobs(classifier, _real_frame_blob_list())
    detections = classifier.classify(warped_bgr, mask_prob, heatmap_prob)

    assert not calibrator.is_calibrated
    by_slot = {(round(d.x_mm), round(d.y_mm)): d for d in detections}
    # Same real documented failure as test_marker_classifier_manifest.py's
    # test_real_logged_hsv_failure_is_reproduced_without_manifest(), now via
    # the adaptive_calibrator-equipped classifier before it has warmed up.
    assert by_slot[(0, -30)].color == "red"  # true black, WRONG (pre-warm-up)
    assert by_slot[(0, -30)].classification_source == "hsv_bins"
    assert by_slot[(0, 30)].color == "yellow"  # true green, WRONG (pre-warm-up)
    assert by_slot[(0, 30)].classification_source == "hsv_bins"


def test_calibrator_fixes_the_real_logged_failure_once_warmed_up() -> None:
    """THE headline integration case: replay the same real frame enough
    times to satisfy warm-up, then confirm every one of the 5 real markers
    (including the two static bins get wrong) resolves correctly via
    classification_source="adaptive_calibration", never reading
    hardware/platform_templates/ at any point."""
    calibrator = AdaptiveColorCalibrator(min_samples_per_slot=3, min_slots_for_calibration=2)
    classifier = MarkerClassifier(input_size=_INPUT_SIZE, adaptive_calibrator=calibrator)

    detections = []
    for _ in range(5):
        warped_bgr, mask_prob, heatmap_prob = _build_frame_with_blobs(classifier, _real_frame_blob_list())
        detections = classifier.classify(warped_bgr, mask_prob, heatmap_prob)

    assert calibrator.is_calibrated
    by_slot = {(round(d.x_mm), round(d.y_mm)): d for d in detections}

    expected_by_slot = {
        (0, -30): "black",
        (30, 0): "yellow",
        (0, 0): "blue",
        (-30, 0): "red",
        (0, 30): "green",
    }
    for slot, expected_color in expected_by_slot.items():
        det = by_slot[slot]
        assert det.color == expected_color, f"slot {slot}: expected {expected_color}, got {det.color}"
        assert det.classification_source == "adaptive_calibration"


def test_manifest_position_still_takes_priority_over_adaptive_calibration() -> None:
    """If a caller supplies BOTH manifest_path (eval/training only) and an
    adaptive_calibrator, the ground-truth manifest tier must still win --
    this test only pins down that existing priority ordering is unchanged,
    it is not an endorsement of combining them in live operation (manifest
    tier stays restricted to evaluation/training call sites, see
    manifest_path's own docstring)."""
    manifest_01 = (
        Path(__file__).resolve().parents[3]
        / "hardware"
        / "platform_templates"
        / "aruco_markers_01_manifest.json"
    )
    calibrator = AdaptiveColorCalibrator(min_samples_per_slot=3, min_slots_for_calibration=2)
    classifier = MarkerClassifier(
        input_size=_INPUT_SIZE, manifest_path=manifest_01, adaptive_calibrator=calibrator
    )

    for _ in range(5):
        warped_bgr, mask_prob, heatmap_prob = _build_frame_with_blobs(classifier, _real_frame_blob_list())
        detections = classifier.classify(warped_bgr, mask_prob, heatmap_prob)

    by_slot = {(round(d.x_mm), round(d.y_mm)): d for d in detections}
    assert by_slot[(0, -30)].classification_source == "manifest_position"
    assert by_slot[(0, -30)].color == "black"
