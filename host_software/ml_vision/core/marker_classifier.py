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
#
# 2026-09-15 recalibration (docs/PROJECT_LOGBOOK.md, "Marker Classifier: 2 of 5
# Colors Unreachable on Jetson AGX Orin"): green/yellow's shared H=35 boundary
# and black's V<60 bound were both re-measured against real photos of the
# aruco_markers_03 sheet (session_20260810_114330, the one real session whose
# masks match this manifest -- 12,883 frames, ~800-1000 sampled per feature):
#   - green_hexagon's real hue (eroded-core, per-frame mean) is 45-85, median
#     ~55, p5~51 -- it essentially never legitimately reads below ~45. yellow_square's
#     real hue runs up to ~42-44 at the high end (full/CNN-mask-realistic
#     population, not the tightest possible erosion) before saturation drops off.
#     The old touching boundary (both at H=35) sat squarely inside yellow's own
#     real upper tail, well below green's real floor -- any yellow reading that
#     drifted up (which real ones routinely do) was guaranteed to hit green's
#     bin first under the old check order. Moved green's floor to 45 and
#     yellow's ceiling to 42, leaving a small 42-45 gap that neither color's
#     real distribution legitimately claims (a value landing there is safer
#     left "unknown" than confidently assigned to the wrong color).
#   - black_circle's real printed marker measures S~11-26, V~110-140 (core
#     pixels, eroded mask, n>35k) even fully unoccluded by the ball -- a
#     desaturated mid-grey under this camera/lighting, NOT anywhere near V<60.
#     Blank background paper (sampled well clear of every marker/ArUco fiducial)
#     measures V~214-255 the great majority of the time, comfortably above 150
#     -- widening black's V bound to 150 covers the marker's measured range
#     (p99=140) with a 10-unit margin while staying well clear of background.
#     NOTE: this does NOT explain the live-Jetson report of black_circle
#     producing zero blobs at all -- run_cnn_mask_check.py against this same
#     session shows shared_vision_backbone_v2's mask head fires at >0.98
#     confidence over black_circle's location in 98%+ of unoccluded frames, so
#     a blob does form here on historical data. This bin fix only helps once a
#     blob exists; if Jetson genuinely produces no blob at all, that is a
#     separate, still-open question (see logbook) needing a live capture to
#     resolve, not something this file can fix.
#   - Widening black's V bound alone (first pass) measurably increased
#     collateral misclassification of the other 4 markers' own dim/shadowed
#     frames as "black" -- verified end-to-end via marker_classifier.classify()
#     itself (not just the raw bins) against 307 real sampled frames: 46 blobs
#     belonging to blue_triangle/yellow_square/green_hexagon/red_triangle
#     flipped from a (harmless) "unknown" to a wrong "black". Their S values
#     (median 29.2, extending down to 13.6) meaningfully overlap black_circle's
#     own real S distribution (median 13.8, p95 27.2, max 31.5 across the same
#     307 frames) -- a real ambiguity in HSV space, not fully separable. Capping
#     black's S at 20 was the best real trade-off measured: keeps 92.8% of true
#     black_circle detections while rejecting 82.6% of the false positives (46
#     -> 8 leaking through). The residual ~8/1228 (~0.65%) false-"black" rate on
#     the other markers' dimmest frames is a known, quantified, NOT fully
#     eliminated residual risk -- tightening further trades away real
#     black_circle recall for diminishing false-positive reduction (see
#     black_s_cap_tradeoff numbers in the 2026-09-15 logbook entry).
COLOR_BINS = {
    "blue": [(np.array([90, 50, 50]), np.array([150, 255, 255]))],
    "red": [
        (np.array([0, 50, 50]), np.array([15, 255, 255])),
        (np.array([165, 50, 50]), np.array([180, 255, 255])),
    ],
    # S floor lowered 50->20 (2026-09-15 follow-up): green_hexagon's real
    # per-frame mean saturation (the value _classify_color() actually sees,
    # over the CNN's full blob, not just an eroded core) sits mostly BELOW
    # even the 45 previously suggested -- swept 15/20/25/30/35/40/45/50 S
    # floors end-to-end via marker_classifier_green_fallback_sweep.py against
    # 1200 real sampled frames from session_20260810_114330: green_hexagon
    # recall was 0.0% (S=50, shipped baseline) / 0.3% (45) / 1.1% (40) / 3.4%
    # (35) / 42.2% (30) / 92.9% (25) / 94.5% (20) / 94.7% (15) -- a sharp
    # inflection between 30 and 20, then a flat plateau (20->15 buys only
    # +0.2pp). Chose 20, not 15, because the plateau's marginal recall gain
    # isn't worth its collateral cost: lowering 20->15 increased false-"green"
    # leakage into black_circle's own real frames from 14/1200 to 21/1200
    # (+50% relative) for +0.2pp of green recall. At S=20 specifically, the
    # other 4 colors' own CORRECT classification counts are byte-for-byte
    # unchanged from the S=50 baseline (black 1124/1200, blue 18/1181, yellow
    # 1033/1198, red 44/1196) -- guaranteed by construction, not just
    # empirically, since blue/red/yellow are checked before green in
    # COLOR_CHECK_ORDER and can only reach green's bin once already rejected
    # by their own. The real, quantified residual (analogous to the black
    # S-cap tradeoff below): S=20 does increase each other color's *false*-
    # green tail relative to the old S=50 bin -- black_circle 0->14/1200
    # (1.2%), yellow_square 52->78/1198 (+2.2 points, since yellow's dim tail
    # already leaked into green somewhat even at S=50), red_triangle
    # 359->375/1196 (+1.3 points, on top of the separate, already-documented
    # red/blue hue-wraparound bug below -- this is a small amplification of
    # that pre-existing bug, not a new one), blue_triangle 0->9/1181 (0.8%).
    # All previously "unknown" blobs turning into a specific wrong color, not
    # previously-correct classifications being overwritten.
    "green": [(np.array([45, 20, 50]), np.array([85, 255, 255]))],
    "yellow": [(np.array([20, 50, 50]), np.array([42, 255, 255]))],
    "black": [(np.array([0, 0, 0]), np.array([180, 20, 150]))],
}
# Check order matters: black's V bound can overlap the low-V edge of the hue
# bins above, so hue-specific colors are checked first and black is the
# fallback -- matches MarkerTracker's existing first-match-wins pattern.
# yellow is now checked before green (was: green, yellow) -- real yellow's
# hue drifts up into the old shared-boundary zone far more often than real
# green's hue drifts down into it (see recalibration note above), so on any
# residual ambiguity it's the safer default.
COLOR_CHECK_ORDER = ["blue", "red", "yellow", "green", "black"]

# Canonical hue centers (OpenCV 0-180 scale) for the nearest-hue fallback tier
# (2026-09-15 follow-up, docs/PROJECT_LOGBOOK.md). Used ONLY when no strict
# COLOR_BINS entry matches -- this is a second, lower-confidence tier, never
# an override of a confident bin match. "black" is deliberately excluded:
# it's an achromatic/low-saturation color where "hue" is close to
# meaningless/noisy (real black_circle pixels' hue is essentially unconstrained
# scatter, not a stable value the way the other 4 colors' hues are) -- the
# empirical sweep in marker_classifier_green_fallback_sweep.py confirmed
# including it produces noisy hue collisions with no corresponding recall
# benefit (black already has its own dedicated S/V bin, which is the correct
# tier for it). Centers are each bin's midpoint hue.
COLOR_HUE_CENTERS = {
    "blue": 120.0,
    "red": 0.0,  # red wraps the hue circle; 0 == 180 on this scale
    "yellow": 31.0,
    "green": 65.0,
}


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


def _circular_hue_distance(h1: float, h2: float, hue_range: float = 180.0) -> float:
    """Wrap-aware distance between two OpenCV hues (0-180 scale, wraps at
    180->0). NOT the same mistake as the pre-existing, out-of-scope
    cv2.mean()-over-a-wrapping-blob bug documented above (that one averages
    raw hue values linearly before this function ever sees them -- e.g. a
    blob straddling the wrap can report a mean hue of ~90 for a true red
    blob). This function only fixes the DISTANCE calculation from an already
    -computed (possibly still wrap-contaminated) mean hue to a canonical
    color center -- e.g. naive `abs(178 - 2)` would report 176 (nearly
    maximally far) when the true circular distance is 4."""
    diff = abs(h1 - h2) % hue_range
    return min(diff, hue_range - diff)


def _nearest_hue_fallback(h: float, threshold: float) -> str:
    """Second-tier color assignment for blobs that hit no strict COLOR_BINS
    entry: nearest COLOR_HUE_CENTERS entry by circular hue distance, IF within
    `threshold`. Returns "unknown" if nothing is close enough (threshold<=0
    disables the tier entirely, since no real distance can be <= 0)."""
    if threshold <= 0:
        return "unknown"
    best_color, best_dist = "unknown", float("inf")
    for color, center in COLOR_HUE_CENTERS.items():
        d = _circular_hue_distance(h, center)
        if d < best_dist:
            best_dist = d
            best_color = color
    if best_dist <= threshold:
        return best_color
    return "unknown"


def _classify_color(mean_hsv: Tuple[float, float, float], fallback_hue_threshold: float = 0.0) -> str:
    h, s, v = mean_hsv
    for color in COLOR_CHECK_ORDER:
        for lower, upper in COLOR_BINS[color]:
            if lower[0] <= h <= upper[0] and lower[1] <= s <= upper[1] and lower[2] <= v <= upper[2]:
                return color
    # Second tier: only reached when no strict bin matched above -- never
    # overrides a confident classification. See COLOR_HUE_CENTERS' comment for
    # why "black" doesn't participate.
    return _nearest_hue_fallback(h, fallback_hue_threshold)


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
        # 2026-09-15 follow-up: nearest-hue fallback tier threshold, chosen by
        # sweeping 0/5/8/10/12/15/20/25/30 end-to-end (marker_classifier_
        # green_fallback_sweep.py --sweep-fallback) against 1200 real sampled
        # frames from session_20260810_114330, on top of the S=20 green bin
        # above. Key finding: for green_hexagon ITSELF this tier recovers
        # ZERO additional detections at ANY threshold tested (its correct
        # count stays flat at 1126/1191 from threshold=0 through 30) --
        # green's residual ~61-frame "unknown" pool's real hue sits in the
        # 42-45 gap deliberately left unresolved by the yellow/green boundary
        # fix above, so nearest-hue matching resolves it toward YELLOW's
        # center (31), not green's (65), once the threshold is wide enough to
        # reach it. There is a sharp cliff at threshold 12->15 where the
        # fraction of green_hexagon's own unknown pool converted to a wrong
        # label jumps from 24.6% to 98.4% with no green recall benefit
        # whatsoever -- i.e. widening past ~12 only trades away green's
        # harmless "unknown" safety net for nothing. Chose 10 (safely below
        # that cliff) because it still captures a real, separate, genuine
        # win this tier IS suited for: blue_triangle and yellow_square have
        # the same "CNN detects a blob, HSV bin fails on desaturation" failure
        # shape as green did, and their real hue IS tightly clustered near
        # their own bin's center (blue: mean circ-dist to center 10.4 +-4.6;
        # yellow: 6.8 +-7.7 -- see the hue-stats diagnostic), so a small
        # threshold correctly recovers many of them: blue_triangle 1.5%->
        # 50.2%, yellow_square 86.2%->86.6%. Quantified cost at threshold=10:
        # black_circle gains 15/1200 (1.25%) new false labels it didn't have
        # with the fallback tier off (10 as blue, 5 as yellow) -- "black" is
        # deliberately NOT a candidate in COLOR_HUE_CENTERS (see its own
        # comment) precisely because black_circle's own real hue (mean 116.6,
        # std 16.3 -- diagnosed via the same hue-stats tool) sits almost on
        # top of blue's canonical center (120, circular distance ~3.4), so
        # black's real detections are themselves at highest risk of being
        # mis-pulled toward "blue" by ANY hue-distance fallback -- a risk
        # that grows with threshold (black-as-false-blue: 0 at threshold<5,
        # 10 at 10, 17 at 15, 45 at 30) and is the main reason NOT to push
        # this threshold higher for marginal extra blue/yellow recall.
        # black_circle's/red_triangle's own CORRECT counts are unaffected at
        # threshold=10 (1124/1200 and 44/1196, byte-for-byte unchanged from
        # fallback-disabled) -- verified, not assumed, same as the S-floor
        # check above.
        fallback_hue_threshold: float = 10.0,
    ) -> None:
        self.input_size = input_size
        self.mask_threshold = mask_threshold
        self.min_blob_area_px = min_blob_area_px
        self.fallback_hue_threshold = fallback_hue_threshold

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
            color = _classify_color(mean_hsv, fallback_hue_threshold=self.fallback_hue_threshold)
            shape = _classify_shape(contour, area)
            cx_px, cy_px = _soft_argmax_centroid(heatmap_prob, blob_mask)
            x_mm, y_mm = self._px_to_touch_mm(cx_px, cy_px)

            detections.append(MarkerDetection(color=color, shape=shape, x_mm=x_mm, y_mm=y_mm, area_px=area))

        return detections
