from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import math

# Share the one calibrated black HSV bin with marker_classifier.py instead of
# duplicating a second, driftable copy of the same number (2026-10-07 fix).
# marker_classifier.py's 2026-09-15 recalibration note measured the real
# printed black marker at S~11-26, V~110-140 (a desaturated mid-grey under
# this project's lighting) and set COLOR_BINS["black"] = S<=20, V<=150
# accordingly. This file's own black bound (V<=60, S<=255) was never updated
# to match and detected zero black blobs on real frames as a result (see
# host_software/ml_jetson_vla/reports/black_marker_occlusion_2026_10_07.md).
from .marker_classifier import COLOR_BINS as _CLASSIFIER_COLOR_BINS

_BLACK_LOWER, _BLACK_UPPER = _CLASSIFIER_COLOR_BINS["black"][0]


class MarkerTracker:
    def __init__(self) -> None:
        # Default robust HSV thresholds for the 4 marker colors.
        # Format: [lower_bound, upper_bound]
        #
        # NOTE on check order (2026-10-07): "black" must be checked before
        # "grey" in find_targets()'s per-blob loop below. The calibrated black
        # bound's S<=20 window sits entirely inside grey's own S<=50 window,
        # and black's V<=150 range mostly overlaps grey's V 30-180 range --
        # with "grey" checked first (the original order), every real black
        # marker blob would have matched "grey" before "black" was ever
        # tested, silently swallowing the V-bound fix above. No real printed
        # sheet in hardware/platform_templates/*_manifest.json defines a grey
        # marker feature (checked: 01/02/03 only define black/blue/green/
        # yellow/red/circle/square/triangle/hexagon), so this reorder has no
        # known collateral cost against real data -- "grey" here is a
        # vestigial 5th slot, not something any current sheet exercises.
        self.hsv_ranges: Dict[str, List[np.ndarray]] = {
            "blue": [np.array([90, 50, 50]), np.array([150, 255, 255])],
            "black": [_BLACK_LOWER.copy(), _BLACK_UPPER.copy()],
            "grey": [np.array([0, 0, 30]), np.array([180, 50, 180])],
            "red_1": [np.array([0, 50, 50]), np.array([15, 255, 255])],
            "red_2": [np.array([165, 50, 50]), np.array([180, 255, 255])],
        }
        self.color_keys = ["blue", "black", "grey", "red_1", "red_2"]
        self.current_color_idx = 0
        self.window_name = "Target Marker Tuning"
        self.tuning_enabled = False

    def setup_tuning_window(self):
        """Initializes a CV2 window with trackbars for real-time HSV tuning."""
        self.tuning_enabled = True
        cv2.namedWindow(self.window_name)

        def nothing(x):
            pass

        cv2.createTrackbar("Color ID", self.window_name, 0, 4, self._on_color_change)
        cv2.createTrackbar("H Min", self.window_name, 0, 179, nothing)
        cv2.createTrackbar("S Min", self.window_name, 0, 255, nothing)
        cv2.createTrackbar("V Min", self.window_name, 0, 255, nothing)
        cv2.createTrackbar("H Max", self.window_name, 179, 179, nothing)
        cv2.createTrackbar("S Max", self.window_name, 255, 255, nothing)
        cv2.createTrackbar("V Max", self.window_name, 255, 255, nothing)

        self._on_color_change(0)

    def _on_color_change(self, idx):
        self.current_color_idx = idx
        color_key = self.color_keys[idx]
        lower, upper = self.hsv_ranges[color_key]

        cv2.setTrackbarPos("H Min", self.window_name, lower[0])
        cv2.setTrackbarPos("S Min", self.window_name, lower[1])
        cv2.setTrackbarPos("V Min", self.window_name, lower[2])
        cv2.setTrackbarPos("H Max", self.window_name, upper[0])
        cv2.setTrackbarPos("S Max", self.window_name, upper[1])
        cv2.setTrackbarPos("V Max", self.window_name, upper[2])

    def read_tuning_window(self):
        """Reads the current trackbar positions and updates the active color range."""
        if not self.tuning_enabled:
            return

        color_key = self.color_keys[self.current_color_idx]
        h_min = cv2.getTrackbarPos("H Min", self.window_name)
        s_min = cv2.getTrackbarPos("S Min", self.window_name)
        v_min = cv2.getTrackbarPos("V Min", self.window_name)
        h_max = cv2.getTrackbarPos("H Max", self.window_name)
        s_max = cv2.getTrackbarPos("S Max", self.window_name)
        v_max = cv2.getTrackbarPos("V Max", self.window_name)

        self.hsv_ranges[color_key][0] = np.array([h_min, s_min, v_min])
        self.hsv_ranges[color_key][1] = np.array([h_max, s_max, v_max])

    def _check_color_match(self, mean_hsv, color_name):
        h, s, v = mean_hsv
        if color_name == "red":
            r1_l, r1_u = self.hsv_ranges["red_1"]
            r2_l, r2_u = self.hsv_ranges["red_2"]
            in_r1 = (
                (r1_l[0] <= h <= r1_u[0])
                and (r1_l[1] <= s <= r1_u[1])
                and (r1_l[2] <= v <= r1_u[2])
            )
            in_r2 = (
                (r2_l[0] <= h <= r2_u[0])
                and (r2_l[1] <= s <= r2_u[1])
                and (r2_l[2] <= v <= r2_u[2])
            )
            return in_r1 or in_r2
        else:
            lower, upper = self.hsv_ranges[color_name]
            return (
                (lower[0] <= h <= upper[0])
                and (lower[1] <= s <= upper[1])
                and (lower[2] <= v <= upper[2])
            )

    def find_targets(
        self, warped_frame: Optional[np.ndarray]
    ) -> Tuple[Dict[str, Optional[Tuple[int, int]]], Dict[str, Any]]:
        # Dict iteration order below IS the color check-priority order (see
        # the __init__ note) -- "black" must stay before "grey".
        targets: Dict[str, Optional[Tuple[int, int]]] = {
            "blue": None,
            "black": None,
            "grey": None,
            "red": None,
        }
        masks: Dict[str, Any] = {}

        if warped_frame is None or warped_frame.size == 0:
            return targets, masks

        # 1. Shape Extraction (Grayscale + Adaptive Threshold)
        gray = cv2.cvtColor(warped_frame, cv2.COLOR_BGR2GRAY)
        hsv_full = cv2.cvtColor(warped_frame, cv2.COLOR_BGR2HSV)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Adaptive Threshold captures contrast edges well
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 5
        )
        edges = cv2.Canny(blurred, 50, 150)

        # Combine threshold and edges for a solid contour mask
        combined_mask = cv2.bitwise_or(thresh, edges)
        kernel = np.ones((3, 3), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

        # 2. Geometric Filtering
        contours, _ = cv2.findContours(
            combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        circular_blobs = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50 or area > 5000:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            circularity = 4 * math.pi * (area / (perimeter * perimeter))

            # The markers are round but not perfect, relaxed threshold to 0.5
            if circularity > 0.5:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    circular_blobs.append((cnt, cx, cy))

        # 3. Color Classification
        for cnt, cx, cy in circular_blobs:
            blob_mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(blob_mask, [cnt], -1, 255, -1)

            # Calculate the mean HSV color inside the circular blob
            mean_color = cv2.mean(hsv_full, mask=blob_mask)[:3]

            for color_name in targets.keys():
                if targets[color_name] is None:
                    if self._check_color_match(mean_color, color_name):
                        targets[color_name] = (cx, cy)
                        break

        return targets, masks
