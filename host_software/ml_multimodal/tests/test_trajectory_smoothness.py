"""Tests for `ml_multimodal/evaluations/trajectory_smoothness.py` (2026-10-13).

Covers: an analytic trajectory with a known constant jerk (so the numerical-gradient
chain can be checked against ground truth, not just "runs without crashing"); a
too-short segment and a non-monotonic-timestamp segment both degrading to `None` rather
than crashing or returning a garbage number; `jerk_metrics_for_segments()`'s
accumulation/exclusion-counting shape; and a real verification against one actual
logged Track 4 telemetry CSV (`data/01_bronze/session_jetson_track4_20260915_151627/
telemetry.csv`) -- loaded and segmented through `evaluate_system_control.py`'s own
`load_telemetry()`/`segment_by_target()` (imported read-only; see this module's own
docstring on why the jerk implementation itself lives in ml_multimodal, not there), so
this reports a real computed number against real data, not a placeholder.

Run (stdlib unittest, no new dependency):
    C:/Users/Admin/.conda/envs/ball_balance_env/python.exe host_software/ml_multimodal/tests/test_trajectory_smoothness.py
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

_THIS_DIR: str = os.path.dirname(os.path.abspath(__file__))
_ML_MULTIMODAL_DIR: str = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_HOST_SOFTWARE_DIR: str = os.path.abspath(os.path.join(_ML_MULTIMODAL_DIR, ".."))
if _HOST_SOFTWARE_DIR not in sys.path:
    sys.path.append(_HOST_SOFTWARE_DIR)

from ml_multimodal.evaluations.trajectory_smoothness import (  # noqa: E402
    MIN_SAMPLES_FOR_JERK,
    jerk_metrics_for_segments,
    trajectory_jerk_mm_s3,
)

_REAL_SESSION_CSV: str = os.path.join(
    _HOST_SOFTWARE_DIR, "data", "01_bronze", "session_jetson_track4_20260915_151627", "telemetry.csv"
)


class AnalyticTrajectoryTests(unittest.TestCase):
    def test_cubic_position_gives_known_constant_jerk(self) -> None:
        """x(t) = t^3, y(t) = 0 has an exact, constant analytic jerk of
        d^3/dt^3(t^3) = 6 mm/s^3 (no acceleration/jerk contribution from y at all). A
        60-sample uniform grid over t in [0, 3]s was checked by hand during development:
        edge_order=2 keeps both mean and RMS within ~5% of the true value 6.0 even with
        this few samples -- real trial segments run hundreds of samples, so this is
        already closer to a worst case than a best case for the implementation."""
        t_s = np.linspace(0.0, 3.0, 60)
        times_ms = t_s * 1000.0
        x_mm = t_s ** 3
        y_mm = np.zeros_like(t_s)

        result = trajectory_jerk_mm_s3(times_ms, x_mm, y_mm)
        self.assertIsNotNone(result)
        mean_abs_jerk, rms_jerk = result
        self.assertAlmostEqual(mean_abs_jerk, 6.0, delta=0.5)
        self.assertAlmostEqual(rms_jerk, 6.0, delta=0.5)

    def test_constant_position_gives_near_zero_jerk(self) -> None:
        """A ball sitting perfectly still (constant touch_x/touch_y) has zero velocity,
        zero acceleration, zero jerk -- the "perfectly smooth" baseline case."""
        times_ms = np.linspace(0.0, 1000.0, 20)
        x_mm = np.full(20, 5.0)
        y_mm = np.full(20, -3.0)

        result = trajectory_jerk_mm_s3(times_ms, x_mm, y_mm)
        self.assertIsNotNone(result)
        mean_abs_jerk, rms_jerk = result
        self.assertAlmostEqual(mean_abs_jerk, 0.0, delta=1e-9)
        self.assertAlmostEqual(rms_jerk, 0.0, delta=1e-9)

    def test_noisier_trajectory_has_higher_jerk_than_smooth_one(self) -> None:
        """Sanity check that the metric actually orders "smooth" vs. "jerky" the right
        way around -- a trajectory with per-sample alternating noise added on top of a
        smooth linear motion must score a higher jerk than the smooth motion alone."""
        t_s = np.linspace(0.0, 2.0, 60)
        times_ms = t_s * 1000.0
        smooth_x = 10.0 * t_s  # constant velocity -- zero jerk in the continuous limit
        rng = np.random.default_rng(0)
        noisy_x = smooth_x + rng.normal(scale=0.5, size=t_s.shape)
        y_mm = np.zeros_like(t_s)

        smooth_result = trajectory_jerk_mm_s3(times_ms, smooth_x, y_mm)
        noisy_result = trajectory_jerk_mm_s3(times_ms, noisy_x, y_mm)
        self.assertIsNotNone(smooth_result)
        self.assertIsNotNone(noisy_result)
        self.assertLess(smooth_result[1], noisy_result[1])  # RMS jerk: smooth < noisy


class DegradesGracefullyTests(unittest.TestCase):
    def test_too_short_segment_returns_none(self) -> None:
        n = MIN_SAMPLES_FOR_JERK - 1
        times_ms = np.linspace(0.0, 100.0, n)
        result = trajectory_jerk_mm_s3(times_ms, np.zeros(n), np.zeros(n))
        self.assertIsNone(result)

    def test_minimum_length_segment_does_not_crash(self) -> None:
        n = MIN_SAMPLES_FOR_JERK
        times_ms = np.linspace(0.0, 100.0, n)
        result = trajectory_jerk_mm_s3(times_ms, np.arange(n, dtype=float), np.zeros(n))
        self.assertIsNotNone(result)

    def test_non_monotonic_timestamps_returns_none(self) -> None:
        n = 10
        times_ms = np.linspace(0.0, 100.0, n)
        times_ms[5] = times_ms[4]  # duplicate timestamp -- a zero-dt step
        result = trajectory_jerk_mm_s3(times_ms, np.arange(n, dtype=float), np.zeros(n))
        self.assertIsNone(result)

    def test_empty_segment_returns_none(self) -> None:
        result = trajectory_jerk_mm_s3(np.array([]), np.array([]), np.array([]))
        self.assertIsNone(result)


class JerkMetricsForSegmentsTests(unittest.TestCase):
    def test_excludes_too_short_segments_from_the_average_but_counts_them(self) -> None:
        n_long = 30
        times_ms = np.arange(n_long + 2, dtype=float) * 33.3  # ~30Hz-like spacing
        touch_x = np.sin(np.arange(n_long + 2) * 0.1) * 10.0
        touch_y = np.cos(np.arange(n_long + 2) * 0.1) * 10.0
        # Segment 0: too short (2 samples). Segment 1: long enough.
        segments = [(0, 2), (2, n_long + 2)]

        out = jerk_metrics_for_segments(times_ms, touch_x, touch_y, segments)
        self.assertIsNotNone(out["Average_Jerk_Mean_mm_s3"])
        self.assertIsNotNone(out["Average_Jerk_RMS_mm_s3"])
        self.assertEqual(out["Jerk_Trials_Excluded"], 1)
        self.assertIn("mm/s^3", out["Jerk_Caveat"])

    def test_all_segments_too_short_gives_none_not_a_crash(self) -> None:
        times_ms = np.array([0.0, 10.0])
        segments = [(0, 2)]
        out = jerk_metrics_for_segments(times_ms, np.array([0.0, 1.0]), np.array([0.0, 1.0]), segments)
        self.assertIsNone(out["Average_Jerk_Mean_mm_s3"])
        self.assertIsNone(out["Average_Jerk_RMS_mm_s3"])
        self.assertEqual(out["Jerk_Trials_Excluded"], 1)


@unittest.skipUnless(
    os.path.exists(_REAL_SESSION_CSV), f"real Track 4 telemetry CSV not found: {_REAL_SESSION_CSV}"
)
class RealTelemetryVerificationTests(unittest.TestCase):
    """Verifies the metric against real, already-logged telemetry -- not a placeholder.
    Imports `evaluate_system_control.py` read-only (this agent's write boundary excludes
    `host_software/evaluations/`; see this module's own docstring) purely to reuse its
    already-correct `load_telemetry()`/`segment_by_target()`, not to duplicate that
    logic here."""

    def test_real_session_produces_a_real_nonzero_jerk_value(self) -> None:
        sys.path.append(_HOST_SOFTWARE_DIR)
        from evaluations.evaluate_system_control import load_telemetry, segment_by_target

        df = load_telemetry(_REAL_SESSION_CSV)
        segments = segment_by_target(df)
        self.assertGreater(len(segments), 0)

        out = jerk_metrics_for_segments(
            df["timestamp_ms"].to_numpy(dtype=float),
            df["touch_x"].to_numpy(dtype=float),
            df["touch_y"].to_numpy(dtype=float),
            segments,
        )
        print(
            f"[real telemetry] session_jetson_track4_20260915_151627: "
            f"n_segments={len(segments)} excluded={out['Jerk_Trials_Excluded']} "
            f"Average_Jerk_Mean_mm_s3={out['Average_Jerk_Mean_mm_s3']} "
            f"Average_Jerk_RMS_mm_s3={out['Average_Jerk_RMS_mm_s3']}"
        )
        self.assertIsNotNone(out["Average_Jerk_Mean_mm_s3"])
        self.assertIsNotNone(out["Average_Jerk_RMS_mm_s3"])
        self.assertGreater(out["Average_Jerk_Mean_mm_s3"], 0.0)
        self.assertGreaterEqual(out["Average_Jerk_RMS_mm_s3"], out["Average_Jerk_Mean_mm_s3"])
        self.assertLess(out["Jerk_Trials_Excluded"], len(segments))  # at least one trial was usable


if __name__ == "__main__":
    unittest.main()
