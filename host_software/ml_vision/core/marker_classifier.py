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

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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
#
# 13/10/2026 follow-up (docs/PROJECT_LOGBOOK.md, "Marker Classifier: cv2.mean()
# Hue-Wraparound Bug Fixed For Real"): fixing the separate cv2.mean()
# linear-hue-averaging bug (see _circular_mean_hue()) changed what _classify_color()
# actually sees for yellow_square's and red_triangle's own hue -- the 2026-09-15
# yellow ceiling (42) was tuned against the OLD buggy linear mean, which is a
# different number from the correctly-computed circular mean. Re-measured yellow's
# own real circular-mean hue (same 1200-frame real-data method as the 09-15 entry,
# full/CNN-realistic mask, matched-by-position population): p1=7.8, p50=15.1,
# p99=23.7 -- nowhere near the old floor of 20, let alone 35. Separately, real
# red_triangle's own low-end (non-wrapped) samples measure p99=5.8 -- the two
# real distributions sit right next to each other with only a ~2-unit true gap
# (yellow's p1=7.8 vs red's p99=5.8). Narrowed red's lower bound 15->6 and
# lowered yellow's floor 20->9, leaving that small real gap deliberately
# unclaimed (same "safer unknown than confidently wrong" discipline as the
# green/yellow 42-45 gap above) instead of moving the boundary on faith.
#
# Separately, the circular-mean fix alone (before any bin edits) measurably
# regressed green_hexagon 94.5%->91.7% end-to-end -- its own real circular-mean
# hue (same measurement method) has p5=44.2, i.e. ~8% of its real population
# now legitimately sits just below the old 45 floor (this tail was previously,
# coincidentally, pulled up into the 45-85 bin by the linear-mean bug -- a case
# of the bug accidentally compensating for an over-tight floor, not a correct
# classification for the right reason). Swept the floor at 45/44/43/42/40
# end-to-end: green_hexagon recovers 91.7%/95.1%/96.1%/96.4%/96.6% as the floor
# drops, with yellow_square's own correct count byte-for-byte UNCHANGED at
# every value (1154/1198) -- guaranteed by construction, yellow is checked
# before green in COLOR_CHECK_ORDER, so lowering green's floor can only claim
# blobs yellow already rejected. Chose 43, not 40 or 42, to keep a deliberate
# 1-unit gap above yellow's own ceiling (42) rather than touching it exactly --
# diminishing returns past 43 (+2.9pp more green, by touching the boundary
# outright) weren't worth giving up that margin.
COLOR_BINS = {
    "blue": [(np.array([90, 50, 50]), np.array([150, 255, 255]))],
    # Red's lower-wrap bound narrowed 15->7 (13/10/2026, alongside the
    # _circular_mean_hue() fix below): once hue is computed correctly instead
    # of via cv2.mean()'s linear average, real red_triangle's own low-end
    # samples (the ones near H=0, not the ones near H=180) measure p90=3.3,
    # p99=5.8 (n=36/398 real matched blobs, session_20260810_114330) -- never
    # anywhere near the old 15 bound. Real yellow_square, measured the same
    # way, sits at H p1=7.8/p50=15.1/p95=19.3 -- i.e. yellow's entire real
    # range used to be inside red's old 0-15 bin. Before this narrowing, the
    # circular-mean fix alone (tested in isolation) caused yellow_square's
    # own correct-detection rate to collapse 86.6%->2.2% (524/1198 of its own
    # blobs newly satisfying red's full strict H/S/V bin) -- NOT a check-order
    # issue (confirmed by swapping COLOR_CHECK_ORDER with no effect: yellow's
    # blobs were failing yellow's OWN H>=20 floor outright, not being stolen
    # by an earlier check). Narrowing to 7 sits in the real gap between red's
    # p99 (5.8) and yellow's p1 (7.8) -- same gap-leaving discipline as the
    # yellow/green H boundary fix (2026-09-15).
    "red": [
        (np.array([0, 50, 50]), np.array([6, 255, 255])),
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
    # 359->375/1196 (+1.3 points, on top of the separate red/blue
    # cv2.mean()-hue-averaging bug -- see _circular_mean_hue() -- this was a
    # small amplification of that bug while it was still unfixed; it was
    # still present at the time this S-floor sweep was run (2026-09-15), fixed
    # for real 13/10/2026), blue_triangle 0->9/1181 (0.8%).
    # All previously "unknown" blobs turning into a specific wrong color, not
    # previously-correct classifications being overwritten.
    "green": [(np.array([43, 20, 50]), np.array([85, 255, 255]))],
    "yellow": [(np.array([9, 50, 50]), np.array([42, 255, 255]))],
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
    # 13/10/2026 sheet-identity fix (see _load_feature_positions_touch_mm() and
    # MarkerClassifier's manifest_path param below): which tier actually produced
    # `color` for this blob -- "manifest_position" (ground-truth sheet lookup,
    # lighting-invariant) or "hsv_bins" (the original, lighting-sensitive classical-
    # CV path). New field with a fixed default -- every existing consumer
    # (main_onnx_shared_vision_audio.py, run_jetson_standalone.py,
    # small_cnn_grounder.py, ...) only ever reads .color/.shape/.x_mm/.y_mm/.area_px
    # by attribute, never positionally/by field count, so this is additive and does
    # not change behavior for any caller that doesn't look at it.
    classification_source: str = "hsv_bins"


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
    180->0). Originally written only to fix the fallback tier's own
    center-distance calculation, while a separate `cv2.mean()`-over-a-
    wrapping-blob bug in blob HSV extraction was deliberately left unfixed
    (see `_circular_mean_hue()` below, which now fixes that bug for real,
    13/10/2026) -- e.g. naive `abs(178 - 2)` would report 176 (nearly
    maximally far) when the true circular distance is 4."""
    diff = abs(h1 - h2) % hue_range
    return min(diff, hue_range - diff)


def _circular_mean_hue(hue_channel: np.ndarray, blob_mask: np.ndarray, hue_range: float = 180.0) -> float:
    """Wrap-aware mean of a blob's OpenCV hue values (0-180 scale, wraps at
    180->0). Fixes the real bug previously described at the `cv2.mean()` call
    site below as "pre-existing" and "out-of-scope" (13/10/2026,
    docs/PROJECT_LOGBOOK.md -- see "Marker Classifier: cv2.mean() Hue-
    Wraparound Bug Fixed For Real"): a plain scalar average of hue is wrong
    whenever a blob's true color straddles the hue wrap point, because hue is
    an angle, not a linear quantity -- e.g. a blob whose pixels split between
    H~178 and H~2 (a real red blob, noisy near the wrap) linearly averages to
    ~90 (green/cyan), nowhere near its true red hue. The fix: treat each
    pixel's hue as a point on the unit circle (angle = hue * 2*pi/180),
    average the unit vectors (mean sin, mean cos), and take the angle of the
    resulting vector (atan2) as the circular mean -- the standard construction
    for averaging angular/circular data, equivalent to minimizing total
    circular distance to every sample. Confirmed end-to-end against real data
    (1200-frame harness, session_20260810_114330): red_triangle's real blobs
    have 38.1% of pixels <=H10 and 59.2% >=H160 (a genuinely wrap-straddling
    distribution) -- the old linear mean reports ~104.6 (does not reach
    red's bin at all); the circular mean correctly reports ~161.9-177.8,
    squarely inside it."""
    ys, xs = np.nonzero(blob_mask)
    if len(ys) == 0:
        return 0.0
    hues = hue_channel[ys, xs].astype(np.float64)
    angles = hues * (2.0 * math.pi / hue_range)
    mean_sin = float(np.sin(angles).mean())
    mean_cos = float(np.cos(angles).mean())
    mean_angle = math.atan2(mean_sin, mean_cos)
    mean_hue = (mean_angle * hue_range) / (2.0 * math.pi)
    return mean_hue % hue_range


# Minimum mean saturation required before the nearest-hue fallback tier is
# allowed to claim a blob at all (13/10/2026 fix, alongside _circular_mean_hue()
# -- docs/PROJECT_LOGBOOK.md, "Marker Classifier: cv2.mean() Hue-Wraparound Bug
# Fixed For Real"). Rationale, same one the file already uses to justify
# excluding "black" from COLOR_HUE_CENTERS: hue is only a meaningful signal
# once a pixel has real chroma -- for a low-saturation/achromatic blob, even a
# *correctly computed* circular mean hue is still just amplified sensor noise,
# not a stable color reading. Before this gate, a live-Jetson diagnostic
# (`diagnose_small_cnn_grounder_live.py`) showed black_circle blobs reading as
# "red" multiple frames in a row -- reproduced here on the historical harness
# once _circular_mean_hue() was wired in: black_circle's own strict S<=20/V<=150
# bin (its correct tier) already fails to match ~6% of its real blobs for
# reasons unrelated to hue (borderline S/V, pre-existing residual, see the
# black S-cap note above) -- those blobs used to scatter toward whichever of
# blue/green/yellow the fallback tier's circular-hue search happened to land
# nearest, and after the circular-mean fix, land nearest to RED far more often
# for a real, measured reason: black_circle's circular-mean hue has a genuine
# right tail reaching into the 160-180 range (p95=163.4, p99=171.9 on the
# 1200-frame harness) -- well within the fallback threshold's reach of red's
# 0.0 center.
#
# Threshold swept (not assumed) at 20/25/30/35/40 end-to-end via the same
# 1200-frame real-data harness, MarkerClassifier() shipped defaults otherwise:
# black_circle's false-"red" count via the fallback tier falls 19 (S=20,
# fallback unguarded) -> 9 (25) -> 6 (30) -> 1 (35) -> 0 (40) as the gate
# tightens. But blue_triangle's own correct-detection rate (which comes almost
# entirely FROM this fallback tier -- only 1.5% without it) holds nearly flat
# through 30 (51.1% -> 50.8% -> 50.4%) then falls off a cliff at 35->40 (31.5%
# -> 7.7%), because blue_triangle's own real mean saturation (38.5, measured
# via the hue-stats diagnostic) starts getting excluded by the gate itself
# past that point. Chose 30: sits clearly below blue's cliff (cost: -0.7pp
# blue, -1.4pp red_triangle, both real but small) while still cutting black's
# false-red count 19->6 (68% relative reduction) -- the residual 6/1200
# (~0.5%) is a known, quantified, NOT fully eliminated risk, same shape as the
# black S-cap and green S-floor residuals documented elsewhere in this file;
# pushing past 30 trades away blue_triangle's entire real recovery for
# diminishing black-side returns. yellow_square's correct count (1154/1198,
# 96.3%) is unaffected by this parameter across the whole swept range -- it
# no longer depends on the fallback tier at all after the H-floor fix above.
MIN_FALLBACK_SATURATION = 30.0


def _nearest_hue_fallback(h: float, s: float, threshold: float, min_saturation: Optional[float] = None) -> str:
    """Second-tier color assignment for blobs that hit no strict COLOR_BINS
    entry: nearest COLOR_HUE_CENTERS entry by circular hue distance, IF within
    `threshold` AND the blob has enough saturation for hue to be a meaningful
    signal at all (see MIN_FALLBACK_SATURATION). Returns "unknown" if nothing
    is close enough, or if saturation is too low to trust hue in the first
    place (threshold<=0 disables the tier entirely, since no real distance
    can be <= 0). `min_saturation` defaults to the module-level
    MIN_FALLBACK_SATURATION, looked up at call time (not bound as a mutable
    default argument) so tests/sweeps can monkeypatch the module constant."""
    if min_saturation is None:
        min_saturation = MIN_FALLBACK_SATURATION
    if threshold <= 0 or s < min_saturation:
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
    # why "black" doesn't participate, and MIN_FALLBACK_SATURATION's comment
    # for why low-saturation blobs are blocked from this tier entirely.
    return _nearest_hue_fallback(h, s, fallback_hue_threshold)


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


# Default manifest-position match radius (mm). Same "half of the sheet's real
# 30mm inter-marker spacing" convention already used elsewhere for exactly this
# purpose: host_software/src/state_machine.py's _MarkerPositionFilter.jump_threshold_mm
# (15.0) and host_software/ml_vision/experiments/marker_classifier_green_fallback_sweep.py's
# own 15mm nearest-feature match radius -- a real blob at its own marker's
# position should land well inside this, and a different marker's own blob
# (30mm away) should land well outside it, per aruco_markers_01/02/03_manifest.json's
# shared feature layout (see ground_truth_manifest.json / CLAUDE.md "Marker labels").
DEFAULT_MANIFEST_MATCH_RADIUS_MM = 15.0


def _load_feature_positions_touch_mm(manifest_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load one sheet manifest's ground-truth `features` list -- each marker's
    physical color/shape/position, known by design for the currently-mounted
    sheet (see `hardware/platform_templates/*_manifest.json`, CLAUDE.md's
    "Marker labels" rule) -- and convert each entry's `center_mm` into the same
    touch-mm frame `MarkerDetection` reports blobs in, via the identical
    `touch_x = W/2 - manifest_x`, `touch_y = manifest_y - H/2` convention
    `_px_to_touch_mm()` already uses (see this module's docstring).

    This is the real fix for the 13/10/2026 "aruco_markers_01 vs _03, different
    sheet + lighting" failure (docs/PROJECT_LOGBOOK.md): a static marker's color
    is fixed, by design, for the life of a session on a given sheet -- looking it
    up by matched physical slot position is lighting/camera/ink invariant,
    because unlike `_classify_color()` it never inspects a single pixel of the
    live frame at all. The investigation that day confirmed real blob positions
    already match the manifest to <0.5mm even when every HSV bin fails on that
    same frame -- i.e. position identification was already working throughout;
    only color identification needed fixing, and this sidesteps it instead of
    re-tuning more bins.

    Deliberately self-contained (reads the manifest JSON directly) rather than
    importing `data_processing/auto_label_shared_vision.py`'s own
    `load_manifest_full()`: that would be this file's first-ever sibling-
    ml_vision-directory import, and `marker_classifier.py` is consumed from
    several call sites (`main_onnx_shared_vision_audio.py`,
    `run_jetson_standalone.py`, `small_cnn_grounder.py`, ...) whose own sys.path
    setup was not audited as part of this change -- duplicating ~10 lines of
    already-stable manifest schema is a smaller, more contained risk than a new
    cross-directory import silently breaking an untouched call site.
    """
    path = Path(manifest_path)
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    platform_w_mm = float(payload["platform_width_mm"])
    platform_h_mm = float(payload["platform_height_mm"])

    positions: List[Dict[str, Any]] = []
    for feature in payload.get("features", []):
        manifest_x, manifest_y = feature["center_mm"]
        touch_x = platform_w_mm / 2.0 - float(manifest_x)
        touch_y = float(manifest_y) - platform_h_mm / 2.0
        positions.append(
            {
                "name": feature.get("name", feature.get("id", "unknown")),
                "color": str(feature["color"]),
                "shape": feature.get("shape", "circle"),
                "x_mm": touch_x,
                "y_mm": touch_y,
            }
        )
    return positions


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
        # 13/10/2026 sheet-identity fix. Opt-in and None by default -- every
        # existing call site (main_onnx_shared_vision_audio.py,
        # run_jetson_standalone.py, small_cnn_grounder.py, ...) constructs
        # MarkerClassifier() without this argument, so self._manifest_features
        # stays [] and classify() falls through to the exact same _classify_color()
        # HSV path as before, byte-for-byte -- zero behavior change for any
        # caller that doesn't explicitly opt in. See
        # _load_feature_positions_touch_mm()'s docstring for the fix itself.
        manifest_path: Optional[Union[str, Path]] = None,
        manifest_match_radius_mm: float = DEFAULT_MANIFEST_MATCH_RADIUS_MM,
    ) -> None:
        self.input_size = input_size
        self.mask_threshold = mask_threshold
        self.min_blob_area_px = min_blob_area_px
        self.fallback_hue_threshold = fallback_hue_threshold
        self.manifest_match_radius_mm = manifest_match_radius_mm
        self._manifest_features: List[Dict[str, Any]] = (
            _load_feature_positions_touch_mm(manifest_path) if manifest_path is not None else []
        )

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
        blob_infos: List[Dict[str, Any]] = []

        for label in range(1, num_labels):
            area = float(stats[label, cv2.CC_STAT_AREA])
            if area < self.min_blob_area_px:
                continue

            blob_mask = (labels == label).astype(np.uint8)
            contours, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)

            # S/V are scalar (non-circular) quantities -- cv2.mean()'s plain average
            # is correct for them. H is an angle and needs the wrap-aware circular
            # mean (_circular_mean_hue()) instead; using cv2.mean()'s linear average
            # for H here was the real, previously out-of-scope bug (13/10/2026 fix,
            # see _circular_mean_hue()'s docstring and docs/PROJECT_LOGBOOK.md).
            _, mean_s, mean_v, _ = cv2.mean(hsv_full, mask=blob_mask * 255)
            mean_h = _circular_mean_hue(hsv_full[:, :, 0], blob_mask)
            shape = _classify_shape(contour, area)
            cx_px, cy_px = _soft_argmax_centroid(heatmap_prob, blob_mask)
            x_mm, y_mm = self._px_to_touch_mm(cx_px, cy_px)

            blob_infos.append(
                {
                    "area": area,
                    "shape": shape,
                    "x_mm": x_mm,
                    "y_mm": y_mm,
                    "mean_hsv": (mean_h, mean_s, mean_v),
                }
            )

        # Color identification, two tiers:
        #   1. Manifest-position lookup (self._manifest_features, only non-empty when
        #      `manifest_path` was supplied at construction) -- a static marker's color
        #      is fixed, by design, for the whole session on whatever sheet is actually
        #      mounted, so matching this blob's own (x_mm, y_mm) against the active
        #      sheet's known feature slots and reading `color` straight off the manifest
        #      never inspects a single pixel of this frame. Lighting/camera/ink-invariant
        #      by construction -- the real fix for the 13/10/2026 "_01 vs _03, different
        #      sheet + lighting" failure (see _load_feature_positions_touch_mm()).
        #   2. _classify_color()'s HSV bins (original path, unchanged) -- used whenever
        #      no manifest was supplied, or this specific blob doesn't land within
        #      manifest_match_radius_mm of any still-unclaimed slot (wrong sheet loaded,
        #      a spurious/off-position blob, or a real marker reading unusually far off
        #      its true spot that frame).
        #
        # Largest-area-first claim order (only exercised when the manifest tier is
        # active -- with it off, every index's own iteration order is unaffected, so
        # manifest_path=None reproduces the pre-fix behavior exactly) so a small spurious
        # blob can't steal a real marker's slot out from under it before the real, larger
        # blob gets a chance to claim it; each manifest slot is claimed at most once.
        claim_order = (
            sorted(range(len(blob_infos)), key=lambda i: blob_infos[i]["area"], reverse=True)
            if self._manifest_features
            else list(range(len(blob_infos)))
        )
        claimed_feature_indices: set = set()
        detections_by_index: Dict[int, MarkerDetection] = {}

        for i in claim_order:
            info = blob_infos[i]
            color: Optional[str] = None
            source = "hsv_bins"

            if self._manifest_features:
                best_idx: Optional[int] = None
                best_dist = float("inf")
                for fidx, feature in enumerate(self._manifest_features):
                    if fidx in claimed_feature_indices:
                        continue
                    dist = math.hypot(info["x_mm"] - feature["x_mm"], info["y_mm"] - feature["y_mm"])
                    if dist < best_dist:
                        best_dist, best_idx = dist, fidx
                if best_idx is not None and best_dist <= self.manifest_match_radius_mm:
                    color = str(self._manifest_features[best_idx]["color"])
                    claimed_feature_indices.add(best_idx)
                    source = "manifest_position"

            if color is None:
                color = _classify_color(info["mean_hsv"], fallback_hue_threshold=self.fallback_hue_threshold)

            detections_by_index[i] = MarkerDetection(
                color=color,
                shape=info["shape"],
                x_mm=info["x_mm"],
                y_mm=info["y_mm"],
                area_px=info["area"],
                classification_source=source,
            )

        return [detections_by_index[i] for i in range(len(blob_infos))]
