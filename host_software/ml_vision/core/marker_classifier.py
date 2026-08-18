"""Classical CV post-processing for the shared vision backbone's marker outputs.

Implements Component 5 of docs/plans/implementation_plan_shared_backbone_cnn.md:
takes the CNN's segmentation mask + heatmap (NOT a from-scratch classical-CV
blob detector like the older, unused MarkerTracker in marker_tracker.py, which
re-derives blobs via adaptive-threshold + Canny) and classifies each blob's
shape and color.

Color bins are the implementation plan's own table (Component 5), sourced from
MarkerTracker's original thresholds. Note the implementation plan's own mm-scale
formula (`mm_x = cx * (182.5/128)`) is STALE (182.5x147.0 was an earlier,
incorrect platform-dimension estimate) -- per .agents/AGENTS.md, the HSV bin
table is the source of truth here, not that formula. The real conversion (used
below) is the one validated in run_shared_vision_inference_on_dataset.py:
touch_x/touch_y (firmware/PID mm convention, what state_machine.py and the
serial protocol expect) relate to the manifest's own mm frame via
touch_x = W/2 - manifest_mm_x, touch_y = manifest_mm_y - H/2 -- confirmed
against the ball-label point-reflection fix in auto_label_shared_vision.py
(2026-08-12).
"""

import math
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

TOUCHPAD_W_MM = 187.5
TOUCHPAD_H_MM = 142.0
PAPER_MARGIN_MM = 6.0

# (H, S, V) lower/upper bounds. Red wraps the hue circle, so it's two ranges.
COLOR_BINS = {
    "blue": [(np.array([90, 50, 50]), np.array([150, 255, 255]))],
    "red": [
        (np.array([0, 50, 50]), np.array([15, 255, 255])),
        (np.array([165, 50, 50]), np.array([180, 255, 255])),
    ],
    "green": [(np.array([35, 50, 50]), np.array([85, 255, 255]))],
    "yellow": [(np.array([20, 50, 50]), np.array([35, 255, 255]))],
    "black": [(np.array([0, 0, 0]), np.array([180, 255, 60]))],
}
# Check order matters: black's V<60 bound can overlap the low-V edge of the hue
# bins above, so hue-specific colors are checked first and black is the
# fallback -- matches MarkerTracker's existing first-match-wins pattern.
COLOR_CHECK_ORDER = ["blue", "red", "green", "yellow", "black"]


@dataclass
class MarkerDetection:
    color: str
    shape: str
    x_mm: float
    y_mm: float
    area_px: float


def _classify_shape(contour: np.ndarray, area: float) -> str:
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return "circle"
    circularity = 4 * math.pi * (area / (perimeter * perimeter))
    if circularity > 0.75:
        return "circle"

    epsilon = 0.04 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    n = len(approx)
    if n == 3:
        return "triangle"
    if n == 4:
        return "square"
    if n >= 6:
        return "hexagon"
    return "circle"


def _classify_color(mean_hsv: Tuple[float, float, float]) -> str:
    h, s, v = mean_hsv
    for color in COLOR_CHECK_ORDER:
        for lower, upper in COLOR_BINS[color]:
            if lower[0] <= h <= upper[0] and lower[1] <= s <= upper[1] and lower[2] <= v <= upper[2]:
                return color
    return "unknown"


def _soft_argmax_centroid(heatmap: np.ndarray, blob_mask: np.ndarray) -> Tuple[float, float]:
    """Sub-pixel centroid refinement: heatmap-weighted centroid within the
    blob's own mask, per Component 5's spec ("soft-argmax on heatmap within
    blob ROI"). Falls back to the blob's plain centroid if the heatmap has no
    signal there (e.g. heatmap head disagrees with mask head on this blob)."""
    ys, xs = np.nonzero(blob_mask)
    weights = heatmap[ys, xs].astype(np.float64)
    total = weights.sum()
    if total <= 1e-6:
        return float(xs.mean()), float(ys.mean())
    cx = float((xs * weights).sum() / total)
    cy = float((ys * weights).sum() / total)
    return cx, cy


class MarkerClassifier:
    def __init__(
        self,
        input_size: Tuple[int, int] = (128, 128),
        mask_threshold: float = 0.5,
        min_blob_area_px: float = 4.0,
    ) -> None:
        self.input_size = input_size
        self.mask_threshold = mask_threshold
        self.min_blob_area_px = min_blob_area_px

        h, w = input_size
        self._mm_per_px_x = (TOUCHPAD_W_MM + 2 * PAPER_MARGIN_MM) / w
        self._mm_per_px_y = (TOUCHPAD_H_MM + 2 * PAPER_MARGIN_MM) / h

    def _px_to_touch_mm(self, px_x: float, px_y: float) -> Tuple[float, float]:
        manifest_mm_x = -PAPER_MARGIN_MM + px_x * self._mm_per_px_x
        manifest_mm_y = -PAPER_MARGIN_MM + px_y * self._mm_per_px_y
        touch_x = TOUCHPAD_W_MM / 2.0 - manifest_mm_x
        touch_y = manifest_mm_y - TOUCHPAD_H_MM / 2.0
        return touch_x, touch_y

    def classify(
        self,
        warped_frame_bgr: np.ndarray,
        mask_prob: np.ndarray,
        heatmap_prob: np.ndarray,
    ) -> List[MarkerDetection]:
        """
        warped_frame_bgr: (H, W, 3) uint8, the same 128x128 platform-warped frame fed to the model.
        mask_prob:        (H, W) float in [0,1] -- sigmoid(mask_logits), NOT raw logits.
        heatmap_prob:      (H, W) float in [0,1] -- sigmoid(heatmap_logits), NOT raw logits.

        Returns one MarkerDetection per connected component of the thresholded mask.
        """
        binary_mask = (mask_prob > self.mask_threshold).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        if num_labels <= 1:
            return []

        hsv_full = cv2.cvtColor(warped_frame_bgr, cv2.COLOR_BGR2HSV)
        detections: List[MarkerDetection] = []

        for label in range(1, num_labels):
            area = float(stats[label, cv2.CC_STAT_AREA])
            if area < self.min_blob_area_px:
                continue

            blob_mask = (labels == label).astype(np.uint8)
            contours, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)

            mean_hsv = cv2.mean(hsv_full, mask=blob_mask * 255)[:3]
            color = _classify_color(mean_hsv)
            shape = _classify_shape(contour, area)
            cx_px, cy_px = _soft_argmax_centroid(heatmap_prob, blob_mask)
            x_mm, y_mm = self._px_to_touch_mm(cx_px, cy_px)

            detections.append(MarkerDetection(color=color, shape=shape, x_mm=x_mm, y_mm=y_mm, area_px=area))

        return detections
