"""
test_state_machine_auto_hold.py

Regression test for the 13/10/2026 fix to
`TargetStateMachine.maybe_auto_hold()` in `host_software/src/state_machine.py`.

Context (see docs/PROJECT_LOGBOOK.md "Marker Target Jump-Gate", 15/09/2026):
`get_target_coords()` was routed through each color's jump-gated/seeded
`_MarkerPositionFilter` to reject single-frame marker misclassification noise,
but `maybe_auto_hold()`'s own "close enough to auto-latch" distance check was
left reading the raw, unfiltered per-frame `marker_coords` dict directly --
the same noise exposure the rest of the jump-gate fix was built to close.

These tests confirm `maybe_auto_hold()` now uses the same filtered estimate
`get_target_coords()` uses (via `self.marker_filters[...].position`), not the
raw `marker_coords` argument, and that it correctly no-ops while that color's
filter is still unconfirmed (never seeded).
"""

import sys
from pathlib import Path

# Shared `src.` import convention (see CLAUDE.md "Import pattern"):
# host_software/ml_vision/tests/ -> parents[2] == host_software/
_host_software_root = Path(__file__).resolve().parents[2]
if str(_host_software_root) not in sys.path:
    sys.path.insert(0, str(_host_software_root))

from src.state_machine import TargetStateMachine  # noqa: E402


def _seed_filter(sm: TargetStateMachine, color: str, x: float, y: float, n: int = 5) -> None:
    """Feed `n` mutually-consistent detections so the color's
    _MarkerPositionFilter seeds and confirms a position, matching
    _MarkerPositionFilter's default seed_window=5 / seed_consistency_mm=10.0."""
    for _ in range(n):
        sm.update_markers({color: (x, y)})


def test_maybe_auto_hold_uses_filtered_estimate_not_raw_marker_coords() -> None:
    """A single noisy/misclassified raw marker_coords reading must NOT affect
    auto-hold once the color's filtered estimate is confirmed -- the whole
    point of routing through the filter instead of the raw dict."""
    sm = TargetStateMachine()
    sm.current_target_name = "red"

    # Seed red's filter at the real target position (50, 50).
    _seed_filter(sm, "red", 50.0, 50.0)
    assert sm.marker_filters["red"].position is not None
    assert sm.marker_filters["red"].position == (50.0, 50.0)

    # Ball sits right at the real target. Raw marker_coords for "red" is
    # garbage (e.g. a different marker's blob misclassified as red) -- far
    # outside the jump-gate threshold -- so a correct implementation must
    # ignore it and use the filtered (50, 50) estimate instead.
    for _ in range(sm.auto_hold_required_frames):
        sm.maybe_auto_hold(50.0, 50.0, {"red": (999.0, 999.0)})

    assert sm.current_target_name == "hold"
    assert sm.hold_x == 50.0
    assert sm.hold_y == 50.0


def test_maybe_auto_hold_noop_while_filter_unconfirmed() -> None:
    """Before a color's filter has ever seeded (position is None),
    maybe_auto_hold() must no-op (not latch, not crash) regardless of what
    raw marker_coords says -- matches get_target_coords()'s own fallback."""
    sm = TargetStateMachine()
    sm.current_target_name = "green"

    assert sm.marker_filters["green"].position is None

    for _ in range(sm.auto_hold_required_frames + 2):
        sm.maybe_auto_hold(0.0, 0.0, {"green": (0.0, 0.0)})

    assert sm.current_target_name == "green"
    assert sm._on_target_frames == 0


def test_maybe_auto_hold_still_latches_on_genuine_approach() -> None:
    """Positive-path sanity check: once seeded, a ball genuinely dwelling
    within tolerance of the filtered position for enough consecutive frames
    still auto-latches (the fix must not just make this always reject)."""
    sm = TargetStateMachine()
    sm.current_target_name = "blue"
    _seed_filter(sm, "blue", -30.0, 10.0)

    # One frame outside tolerance resets the dwell counter.
    sm.maybe_auto_hold(100.0, 100.0, {"blue": (-30.0, 10.0)})
    assert sm._on_target_frames == 0

    for _ in range(sm.auto_hold_required_frames):
        sm.maybe_auto_hold(-30.0, 10.0, {"blue": (-30.0, 10.0)})

    assert sm.current_target_name == "hold"
    assert sm.hold_x == -30.0
    assert sm.hold_y == 10.0


def test_maybe_auto_hold_resets_on_center_or_hold_target() -> None:
    """Unrelated existing behavior (center/hold targets always no-op) must
    survive the refactor unchanged."""
    sm = TargetStateMachine()
    sm._on_target_frames = 3

    sm.current_target_name = "center"
    sm.maybe_auto_hold(0.0, 0.0, {})
    assert sm._on_target_frames == 0

    sm._on_target_frames = 3
    sm.current_target_name = "hold"
    sm.maybe_auto_hold(0.0, 0.0, {})
    assert sm._on_target_frames == 0


def test_get_target_coords_warns_once_for_a_colour_with_no_filter_at_all() -> None:
    """2026-10-14 fix: "grey" is a real, producible audio command (confirmed via
    audio_receiver_onnx.py's LABEL_NAMES) with no marker_filters entry -- it used to silently
    resolve to permanent center-hold, indistinguishable from the normal "filter exists, not yet
    confirmed" transient. Now logs a one-time warning distinguishing the two cases, via
    `_unfilterable_targets_warned`."""
    sm = TargetStateMachine()
    sm.current_target_name = "grey"
    assert "grey" not in sm.marker_filters  # the real, confirmed-missing case this fix targets

    assert sm._unfilterable_targets_warned == set()
    x, y = sm.get_target_coords()
    assert (x, y) == (0.0, 0.0)  # behavior unchanged -- still falls back to center
    assert sm._unfilterable_targets_warned == {"grey"}

    # Second call for the same target must not re-warn (checked via the tracking set staying
    # a single entry, not growing) -- repeat calls are the normal per-frame case.
    sm.get_target_coords()
    assert sm._unfilterable_targets_warned == {"grey"}


def test_get_target_coords_does_not_warn_for_a_colour_with_a_real_but_unconfirmed_filter() -> None:
    """The normal, NOT-broken case: a real color (has a marker_filters entry) that just hasn't
    been seen/confirmed yet must stay silent -- only the no-filter-at-all case (above) warns."""
    sm = TargetStateMachine()
    sm.current_target_name = "blue"
    assert "blue" in sm.marker_filters  # has a real filter, just unconfirmed

    x, y = sm.get_target_coords()
    assert (x, y) == (0.0, 0.0)
    assert sm._unfilterable_targets_warned == set()  # no warning for this case
