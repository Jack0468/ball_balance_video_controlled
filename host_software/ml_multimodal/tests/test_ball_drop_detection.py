"""Tests for `ml_multimodal/evaluations/ball_drop_detection.py` (2026-10-14).

Covers: the trailing-run/prior-contact logic in isolation with synthetic, exactly-
controlled boundary cases (sub-threshold blip not flagged, over-threshold flagged, the
all-invalid "leading-stale-value" case correctly excluded even though it ends on
touch_valid==0, a segment with no touch_valid column at all degrading to None rather
than crashing); `ball_drop_metrics_for_segments()`'s three-way bucketing and the
settled-but-later-dropped override; and a REAL verification against the 3 genuine drop
candidates and 1 genuine leading-stale false-positive-trap found in this project's own
logged telemetry (`host_software/data/01_bronze/evaluation/
ground_truth_jetson_*.csv`), per the audit in `ball_drop_detection.py`'s own module
docstring -- loaded through `evaluate_system_control.py`'s own `load_telemetry()`/
`segment_by_target()`/`find_settling_time_ms()` (imported read-only; this agent's write
boundary excludes `host_software/evaluations/`), so these report real computed
numbers against real data, not placeholders.

Honesty note (also stated in `ball_drop_detection.py`'s module docstring, repeated here
because it matters for how to read these tests): no session in this repo carries an
authoritative, human-confirmed "a ball drop happened here" label. The "real drop
candidate" tests below confirm the detector fires on the specific real telemetry
pattern the audit found (sustained, non-recovered loss of contact after confirmed
prior contact, far above the observed noise floor) -- they do NOT confirm those 3
historical trials were actually drops in the real world. That confirmation is exactly
the JETSON-HW-gated step this task's own TODO item defers to hardware.

Run (stdlib unittest, no new dependency):
    C:/Users/Admin/.conda/envs/ball_balance_env/python.exe host_software/ml_multimodal/tests/test_ball_drop_detection.py
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

from ml_multimodal.evaluations.ball_drop_detection import (  # noqa: E402
    MIN_DROP_DURATION_MS,
    ball_drop_metrics_for_segments,
    is_likely_ball_drop,
    segment_drop_duration_ms,
)

_EVAL_DIR: str = os.path.join(_HOST_SOFTWARE_DIR, "data", "01_bronze", "evaluation")
_REAL_DROP_CSV: str = os.path.join(_EVAL_DIR, "ground_truth_jetson_20260922_104113.csv")
_REAL_RECOVERABLE_CSV: str = os.path.join(_EVAL_DIR, "ground_truth_jetson_20260922_103659.csv")
_REAL_CLEAN_CSV: str = os.path.join(_EVAL_DIR, "ground_truth_jetson_20260915_151627.csv")


def _uniform_times_ms(n: int, dt_ms: float = 40.0) -> np.ndarray:
    return np.arange(n, dtype=np.float64) * dt_ms


class SegmentDropDurationSyntheticTests(unittest.TestCase):
    def test_no_touch_valid_column_returns_none(self) -> None:
        self.assertIsNone(segment_drop_duration_ms(_uniform_times_ms(10), None))

    def test_still_valid_at_segment_end_returns_none(self) -> None:
        tv = np.array([1, 1, 1, 0, 0, 1])  # recovered before the segment ended
        self.assertIsNone(segment_drop_duration_ms(_uniform_times_ms(6), tv))

    def test_entirely_invalid_segment_returns_none_leading_stale_case(self) -> None:
        """The real `ground_truth_jetson_20260922_103659.csv` segment 0 case: never
        once valid in this segment, so there is no confirmed prior contact to have
        been dropped FROM -- must not be flagged even though it ends on
        touch_valid==0 and the run is long."""
        tv = np.zeros(50, dtype=int)
        self.assertIsNone(segment_drop_duration_ms(_uniform_times_ms(50), tv))

    def test_sub_threshold_trailing_run_returns_none(self) -> None:
        # 1,1,1 valid, then invalid for only 2 samples (80ms at 40ms/sample) -- below
        # MIN_DROP_DURATION_MS=1000ms, ordinary boundary noise.
        tv = np.array([1, 1, 1, 0, 0])
        self.assertIsNone(segment_drop_duration_ms(_uniform_times_ms(5), tv))

    def test_over_threshold_trailing_run_is_flagged_with_correct_duration(self) -> None:
        n_valid = 5
        n_invalid = 30  # 30 * 40ms = 1200ms > MIN_DROP_DURATION_MS
        tv = np.concatenate([np.ones(n_valid, dtype=int), np.zeros(n_invalid, dtype=int)])
        times_ms = _uniform_times_ms(n_valid + n_invalid)
        result = segment_drop_duration_ms(times_ms, tv)
        self.assertIsNotNone(result)
        expected_ms = times_ms[-1] - times_ms[n_valid]
        self.assertAlmostEqual(result, expected_ms, delta=1e-9)
        self.assertGreater(result, MIN_DROP_DURATION_MS)
        self.assertTrue(is_likely_ball_drop(times_ms, tv))

    def test_exact_threshold_boundary(self) -> None:
        """Duration exactly at MIN_DROP_DURATION_MS IS flagged (the reject condition in
        segment_drop_duration_ms is a strict '<', so '==' passes through as a drop);
        one unit below it is NOT."""
        times_ms = np.array([0.0, 500.0, 1000.0, 2000.0])  # invalid run = 1000ms exactly
        tv = np.array([1, 1, 0, 0])
        self.assertIsNotNone(segment_drop_duration_ms(times_ms, tv, min_drop_duration_ms=1000.0))

        times_ms_under = np.array([0.0, 500.0, 1000.0, 1999.0])  # invalid run = 999ms
        self.assertIsNone(segment_drop_duration_ms(times_ms_under, tv, min_drop_duration_ms=1000.0))

    def test_empty_segment_returns_none(self) -> None:
        self.assertIsNone(segment_drop_duration_ms(np.array([]), np.array([])))


class BallDropMetricsForSegmentsTests(unittest.TestCase):
    def test_no_touch_valid_column_degrades_gracefully(self) -> None:
        segments = [(0, 5), (5, 10)]
        out = ball_drop_metrics_for_segments(_uniform_times_ms(10), None, segments, [True, False])
        self.assertIsNone(out["Dropped_Ball_Trials"])
        self.assertIsNone(out["Never_Settled_Trials"])
        self.assertIn("does not carry touch_valid", out["Task_Success_Rate_Caveat"])

    def test_three_way_bucketing_and_drop_overrides_settled(self) -> None:
        # Segment 0: settled, touch stayed valid throughout -> real success.
        tv0 = np.ones(10, dtype=int)
        # Segment 1: never settled, touch stayed valid throughout -> real never-settled.
        tv1 = np.ones(8, dtype=int)
        # Segment 2: settle_ms was non-None (the existing check said "success"), but
        # touch was lost for a long trailing run after confirmed contact -> must be
        # reclassified as Dropped, not Settled, despite settled_mask[2]=True.
        tv2 = np.concatenate([np.ones(5, dtype=int), np.zeros(30, dtype=int)])
        # Segment 3: too short to evaluate at all (settled_mask=None) -> excluded.
        tv3 = np.array([1, 0])

        touch_valid = np.concatenate([tv0, tv1, tv2, tv3])
        segments = [(0, 10), (10, 18), (18, 18 + len(tv2)), (18 + len(tv2), 18 + len(tv2) + 2)]
        settled_mask = [True, False, True, None]
        times_ms = _uniform_times_ms(len(touch_valid))

        out = ball_drop_metrics_for_segments(times_ms, touch_valid, segments, settled_mask)
        self.assertEqual(out["Dropped_Ball_Trials"], 1)
        self.assertEqual(out["Never_Settled_Trials"], 1)
        self.assertEqual(out["Settled_Trials_Overridden_By_Drop"], 1)
        # total_resolved = dropped(1, seg2 reclassified) + never_settled(1, seg1) +
        # corrected_successes(1, seg0) = 3; segment 3 excluded (settled_mask=None).
        self.assertAlmostEqual(out["Task_Success_Rate_Percent_Drop_Corrected"], 100.0 / 3.0)

    def test_settled_and_not_dropped_counts_as_corrected_success(self) -> None:
        tv = np.ones(10, dtype=int)
        segments = [(0, 10)]
        out = ball_drop_metrics_for_segments(_uniform_times_ms(10), tv, segments, [True])
        self.assertEqual(out["Dropped_Ball_Trials"], 0)
        self.assertEqual(out["Never_Settled_Trials"], 0)
        self.assertEqual(out["Settled_Trials_Overridden_By_Drop"], 0)
        self.assertAlmostEqual(out["Task_Success_Rate_Percent_Drop_Corrected"], 100.0)


@unittest.skipUnless(
    os.path.exists(_REAL_DROP_CSV) and os.path.exists(_REAL_RECOVERABLE_CSV) and os.path.exists(_REAL_CLEAN_CSV),
    f"real evaluation telemetry CSVs not found under {_EVAL_DIR}",
)
class RealTelemetryVerificationTests(unittest.TestCase):
    """Verifies the detector against real, already-logged telemetry -- see module
    docstring's honesty note on what this does and does not confirm."""

    def _load_and_segment(self, csv_path: str):
        sys.path.append(_HOST_SOFTWARE_DIR)
        from evaluations.evaluate_system_control import load_telemetry, segment_by_target

        df = load_telemetry(csv_path)
        segments = segment_by_target(df)
        return df, segments

    def test_real_drop_candidate_is_flagged_and_overrides_a_settle(self) -> None:
        """ground_truth_jetson_20260922_104113.csv segment 4: the trial's (and the
        session's) last 257 rows / 10239ms are a non-recovered touch_valid==0 run
        after confirmed earlier contact, AND find_settling_time_ms() independently
        reports this segment as settled (601ms) because the error dipped under
        tolerance before the drop -- exactly the masking case the module docstring
        flags. Confirms both the raw detector and the override."""
        sys.path.append(_HOST_SOFTWARE_DIR)
        from evaluations.evaluate_system_control import (
            SETTLE_DURATION_MS,
            SETTLE_TOLERANCE_MM,
            find_settling_time_ms,
        )

        df, segments = self._load_and_segment(_REAL_DROP_CSV)
        self.assertEqual(len(segments), 5)
        start, end = segments[4]
        times = df["timestamp_ms"].to_numpy(dtype=float)
        touch_valid = df["touch_valid"].to_numpy()
        errors = df["error_mm"].to_numpy(dtype=float)

        seg_times, seg_tv, seg_err = times[start:end], touch_valid[start:end], errors[start:end]
        drop_ms = segment_drop_duration_ms(seg_times, seg_tv)
        settle_ms = find_settling_time_ms(seg_times, seg_err, SETTLE_TOLERANCE_MM, SETTLE_DURATION_MS)
        print(
            f"[real telemetry] ground_truth_jetson_20260922_104113 segment 4: "
            f"drop_ms={drop_ms} settle_ms={settle_ms} (settle_ms non-None here is the "
            f"masking bug this module's override exists to fix)"
        )
        self.assertIsNotNone(drop_ms)
        self.assertGreater(drop_ms, MIN_DROP_DURATION_MS)
        self.assertIsNotNone(settle_ms)  # the pre-existing masking finding, reconfirmed

        settled_mask = [settle_ms is not None]
        out = ball_drop_metrics_for_segments(seg_times, seg_tv, [(0, len(seg_times))], settled_mask)
        self.assertEqual(out["Dropped_Ball_Trials"], 1)
        self.assertEqual(out["Settled_Trials_Overridden_By_Drop"], 1)

    def test_leading_stale_segment_is_not_flagged_as_a_drop(self) -> None:
        """ground_truth_jetson_20260922_103659.csv segment 0: entirely touch_valid==0
        (the pre-recording-stale artifact, not a drop) -- must NOT be flagged despite
        being a long, trailing-to-end-of-segment invalid run."""
        df, segments = self._load_and_segment(_REAL_RECOVERABLE_CSV)
        start, end = segments[0]
        touch_valid = df["touch_valid"].to_numpy()[start:end]
        self.assertTrue(bool(np.all(touch_valid == 0)))  # confirms the premise
        times = df["timestamp_ms"].to_numpy(dtype=float)[start:end]
        self.assertIsNone(segment_drop_duration_ms(times, touch_valid))

    def test_real_recoverable_candidate_also_flagged(self) -> None:
        """Same session's segment 6: a real (if smaller, 2760ms) trailing non-recovered
        loss after confirmed contact -- the second of the 3 real candidates found by
        the audit."""
        df, segments = self._load_and_segment(_REAL_RECOVERABLE_CSV)
        start, end = segments[6]
        touch_valid = df["touch_valid"].to_numpy()[start:end]
        times = df["timestamp_ms"].to_numpy(dtype=float)[start:end]
        drop_ms = segment_drop_duration_ms(times, touch_valid)
        self.assertIsNotNone(drop_ms)
        self.assertGreater(drop_ms, MIN_DROP_DURATION_MS)

    def test_clean_session_flags_zero_drops(self) -> None:
        """ground_truth_jetson_20260915_151627.csv: a session that completed normally
        (no anomalously-short-recording pattern) -- every segment's trailing run
        should be well under MIN_DROP_DURATION_MS, so nothing is flagged."""
        df, segments = self._load_and_segment(_REAL_CLEAN_CSV)
        times = df["timestamp_ms"].to_numpy(dtype=float)
        touch_valid = df["touch_valid"].to_numpy()
        n_flagged = sum(
            1
            for start, end in segments
            if segment_drop_duration_ms(times[start:end], touch_valid[start:end]) is not None
        )
        self.assertEqual(n_flagged, 0)


if __name__ == "__main__":
    unittest.main()
