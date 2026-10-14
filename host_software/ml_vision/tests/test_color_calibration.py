"""
test_color_calibration.py

Unit tests for core/color_calibration.py's AdaptiveColorCalibrator -- the
real, pixel-only fix for docs/TODO.md's "Marker colour classification needs a
robust design, not another hardcoded fix -- REOPENED 2026-10-14" item.

Background: `marker_classifier.py`'s static HSV bins were calibrated once
against `session_20260810_114330` (sheet `aruco_markers_03`) and fail on the
physically-mounted live sheet (`aruco_markers_01`, different sheet + lighting
-- docs/PROJECT_LOGBOOK.md 13/10/2026). A 2026-10-13 fix that looked a blob's
colour up directly from `hardware/platform_templates/*_manifest.json` (ground
truth) was reverted 2026-10-14 by explicit user instruction -- using the
answer key at inference time defeats the point of running a vision pipeline.
This module instead derives a position->colour table purely from the
CURRENT session's own observed camera pixels.

The real logged `aruco_markers_01` HSV values used throughout this file are
not invented -- they are the exact (mean_h, mean_s, mean_v) numbers from
docs/PROJECT_LOGBOOK.md's 13/10/2026 "Live-hardware verification closes..."
results table (same real numbers `test_marker_classifier_manifest.py` already
uses for its own two cases; this file adds the other 3 real markers from that
same table/frame to test all 5 together, which is what the calibrator's
relative/ordinal reasoning actually needs).
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from core.color_calibration import (  # noqa: E402
    MIN_BLACK_V_GAP,
    AdaptiveColorCalibrator,
    _assign_chromatic_labels,
    _rotate_to_largest_gap,
)
from core.color_calibration import _Slot  # noqa: E402


def _slot(h: float, s: float, v: float, x_mm: float = 0.0, y_mm: float = 0.0) -> _Slot:
    slot = _Slot(x_mm=x_mm, y_mm=y_mm)
    slot.update(h, s, v, x_mm, y_mm)
    return slot

# Real, logged aruco_markers_01 frame (docs/PROJECT_LOGBOOK.md 13/10/2026):
# name -> (mean_h, mean_s, mean_v, x_mm, y_mm). Positions per
# aruco_markers_01_manifest.json's confirmed live layout: center=blue,
# top=black, right=red, bottom=green, left=yellow.
REAL_ARUCO_01_FRAME = {
    "black": (7.8, 47.3, 148.2, 0.0, -30.0),
    "yellow": (15.1, 100.9, 234.7, 30.0, 0.0),
    "blue": (129.8, 34.6, 157.9, 0.0, 0.0),
    "red": (1.9, 84.4, 214.8, -30.0, 0.0),
    "green": (34.2, 51.9, 157.6, 0.0, 30.0),
}


def _warm_up(calibrator: AdaptiveColorCalibrator, frame: dict, n_repeats: int) -> None:
    """Feeds the same real per-marker (h, s, v, x, y) readings `n_repeats`
    times -- calibration needs several observations per slot before it
    trusts a mean enough to act on it (MIN_SAMPLES_PER_SLOT); replaying the
    one real captured frame this many times is NOT a substitute for genuine
    multi-frame real data (stated plainly, same as the comparison harness),
    it only satisfies that sample-count gate using real, not synthetic,
    numbers."""
    for _ in range(n_repeats):
        for h, s, v, x, y in frame.values():
            calibrator.observe(h, s, v, x, y)
        calibrator.maybe_calibrate()


def test_not_calibrated_before_enough_slots_or_samples() -> None:
    cal = AdaptiveColorCalibrator(min_samples_per_slot=5, min_slots_for_calibration=2)
    assert not cal.is_calibrated

    # Only 1 slot ever observed -- below min_slots_for_calibration, and well
    # under max_warmup_frames, so calibration must not fire.
    for _ in range(4):
        cal.observe(0.0, 80.0, 200.0, 0.0, 0.0)
        cal.maybe_calibrate()
    assert not cal.is_calibrated
    assert cal.classify_position(0.0, 0.0) is None


def test_calibrates_once_enough_slots_and_samples_exist() -> None:
    cal = AdaptiveColorCalibrator(min_samples_per_slot=3, min_slots_for_calibration=2)
    _warm_up(cal, REAL_ARUCO_01_FRAME, n_repeats=4)
    assert cal.is_calibrated
    assert len(cal.features) == 5


def test_max_warmup_frames_forces_calibration_with_a_permanently_missing_marker() -> None:
    """Matches the 15/09/2026 logbook entry's black_circle-at-dead-center
    scenario: one marker is never detected at all (e.g. permanently occluded).
    Calibration must still eventually fire on whatever slots DO exist, rather
    than waiting forever for a marker that will never appear."""
    cal = AdaptiveColorCalibrator(
        min_samples_per_slot=5, min_slots_for_calibration=10, max_warmup_frames=5
    )
    # Only 3 of the usual 5 markers ever observed (simulating 2 permanently
    # occluded/never-detected markers) -- min_slots_for_calibration=10 can
    # never be satisfied by real observations, so only the frame timeout can
    # trigger calibration.
    partial_frame = {k: REAL_ARUCO_01_FRAME[k] for k in ("black", "red", "blue")}
    for _ in range(6):
        for h, s, v, x, y in partial_frame.values():
            cal.observe(h, s, v, x, y)
        cal.maybe_calibrate()
    assert cal.is_calibrated
    assert len(cal.features) == 3


def test_classify_position_returns_none_before_calibration_and_when_far_away() -> None:
    cal = AdaptiveColorCalibrator(min_samples_per_slot=3, min_slots_for_calibration=2)
    assert cal.classify_position(0.0, 0.0) is None  # not calibrated yet

    _warm_up(cal, REAL_ARUCO_01_FRAME, n_repeats=4)
    assert cal.is_calibrated
    # 50mm from every real slot -- outside slot_match_radius_mm (15.0).
    assert cal.classify_position(50.0, 50.0) is None


def test_reset_clears_all_state() -> None:
    cal = AdaptiveColorCalibrator(min_samples_per_slot=3, min_slots_for_calibration=2)
    _warm_up(cal, REAL_ARUCO_01_FRAME, n_repeats=4)
    assert cal.is_calibrated

    cal.reset()
    assert not cal.is_calibrated
    assert cal.features == []
    assert cal.classify_position(0.0, 0.0) is None


def test_observe_is_a_noop_once_calibrated() -> None:
    """Feeding garbage after calibration must not corrupt the already-
    resolved table -- recalibrating mid-session is explicitly out of scope
    (see observe()'s own docstring); reset() is the deliberate escape hatch."""
    cal = AdaptiveColorCalibrator(min_samples_per_slot=3, min_slots_for_calibration=2)
    _warm_up(cal, REAL_ARUCO_01_FRAME, n_repeats=4)
    assert cal.is_calibrated
    features_before = list(cal.features)

    cal.observe(999.0, 999.0, 999.0, 0.0, 0.0)  # nonsense hue, same position as "black"
    assert cal.features == features_before


def test_real_aruco_01_frame_resolves_all_5_colours_correctly() -> None:
    """THE headline case this fix targets: the exact real logged frame where
    every static HSV bin got black and green wrong (see
    test_marker_classifier_manifest.py's own reproduction of that failure).
    Relative/ordinal reasoning over all 5 real markers together must resolve
    every one correctly, including black (via the real, if modest, V gap)
    and the 4 chromatic colours (via canonical cyclic hue order, despite
    their absolute hues sitting nowhere near their historical _03 centers)."""
    cal = AdaptiveColorCalibrator(min_samples_per_slot=3, min_slots_for_calibration=2)
    _warm_up(cal, REAL_ARUCO_01_FRAME, n_repeats=4)
    assert cal.is_calibrated

    for true_color, (h, s, v, x, y) in REAL_ARUCO_01_FRAME.items():
        assert cal.classify_position(x, y) == true_color, f"{true_color} resolved incorrectly"


def test_black_not_assigned_when_no_marker_is_clearly_darkest() -> None:
    """If every slot's brightness is close together (no genuinely-achromatic
    marker present), nothing should be forced into "black" -- it should fall
    through to the chromatic assignment instead. Uses 4 chromatic slots with
    V values within MIN_BLACK_V_GAP of each other."""
    cal = AdaptiveColorCalibrator(min_samples_per_slot=3, min_slots_for_calibration=2)
    close_v_frame = {
        "red": (1.9, 84.4, 200.0, -30.0, 0.0),
        "yellow": (15.1, 100.9, 205.0, 30.0, 0.0),
        "green": (34.2, 51.9, 202.0, 0.0, 30.0),
        "blue": (129.8, 34.6, 198.0, 0.0, 0.0),
    }
    _warm_up(cal, close_v_frame, n_repeats=4)
    assert cal.is_calibrated
    labels = {f.color for f in cal.features}
    assert "black" not in labels


def test_known_limitation_aruco_markers_02_repeated_colour_not_handled() -> None:
    """DOCUMENTED, NOT-YET-SOLVED limitation (see color_calibration.py's own
    module docstring): aruco_markers_02's 4 shapes are all really "blue"
    (hardware/platform_templates/aruco_markers_02_manifest.json), but this
    calibrator assumes every distinct position slot is a DIFFERENT colour --
    correct for aruco_markers_01/_03 (the only sheets with real captured
    data), wrong for _02. This test pins down the CURRENT (known-incorrect)
    behaviour so a future fix changes a test, not silently drifts unnoticed."""
    cal = AdaptiveColorCalibrator(min_samples_per_slot=3, min_slots_for_calibration=2)
    four_same_blue = {
        "shape_0": (118.0, 60.0, 180.0, 0.0, -30.0),
        "shape_1": (120.0, 62.0, 182.0, 30.0, 0.0),
        "shape_2": (122.0, 58.0, 178.0, 0.0, 30.0),
        "shape_3": (119.0, 61.0, 181.0, -30.0, 0.0),
    }
    _warm_up(cal, four_same_blue, n_repeats=4)
    assert cal.is_calibrated
    labels = [f.color for f in cal.features]
    # Known-wrong: forced into 4 different labels instead of 4x "blue".
    assert len(set(labels)) > 1


def test_chromatic_assignment_with_fewer_than_4_chromatic_slots() -> None:
    """Only 2 markers visible this session (e.g. partial occlusion) -- must
    still pick the best 2-length run of canonical order, not require all 4.
    Uses _assign_chromatic_labels() (the real call path, which rotates to
    the largest gap first) rather than calling the lower-level
    _best_fit_canonical_run() with un-rotated hues directly -- that helper's
    own docstring requires pre-rotated input, and feeding it raw unordered
    hues is not a real usage pattern."""
    slots = [_slot(1.9, 84.4, 214.8), _slot(129.8, 34.6, 157.9)]
    labels = _assign_chromatic_labels(slots)
    assert labels == ["red", "blue"]


def test_chromatic_assignment_with_more_than_4_chromatic_slots() -> None:
    """More distinct chromatic slots than this project has chromatic colours
    (most likely a spurious extra blob holding a stable position) -- must
    label the best-fitting 4 and mark the rest "unknown" rather than
    guessing a 5th real colour."""
    slots = [_slot(h, 80.0, 200.0) for h in (1.9, 15.1, 34.2, 70.0, 129.8)]
    labels = _assign_chromatic_labels(slots)
    assert labels.count("unknown") == 1
    assert len(labels) == 5


def test_rotate_to_largest_gap_handles_the_hue_wrap_correctly() -> None:
    """Two hues straddling the 180->0 wrap (e.g. 2.0 and 178.0) are
    circularly close (4 degrees apart) and must end up ADJACENT in the
    rotated ordering, not at opposite ends of a naive linear sort."""
    hues = [2.0, 178.0, 90.0]
    perm = _rotate_to_largest_gap(hues)
    rotated = [hues[i] for i in perm]
    # The largest real gap is between 90 and 178 (88) and between 2 and 90
    # (88) -- either direction is a valid "cut"; what matters is that 178.0
    # and 2.0 (the wrap-straddling pair) are adjacent in the result.
    idx_178 = rotated.index(178.0)
    idx_2 = rotated.index(2.0)
    assert abs(idx_178 - idx_2) == 1 or {idx_178, idx_2} == {0, len(rotated) - 1}


def test_min_black_v_gap_shipped_default_is_below_the_real_aruco_01_gap() -> None:
    """Regression pin: the real aruco_markers_01 frame's own black-vs-next-
    darkest V gap is 148.2 (black) vs 157.6 (green) = 9.4. The shipped
    default must stay below this or the fix silently stops working for the
    exact case it was built for."""
    real_gap = REAL_ARUCO_01_FRAME["green"][2] - REAL_ARUCO_01_FRAME["black"][2]
    assert MIN_BLACK_V_GAP < real_gap
