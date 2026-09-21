"""Detects corrupted touch-plate ground-truth readings shared by two consumers:
host_software/evaluations/evaluate_system_control.py (live Jetson eval telemetry,
src/touch_logger.py's schema) and
host_software/ml_vision/data_processing/audit_touch_ground_truth_spikes.py (the
older PID-firmware training-session telemetry schema, no touch_valid column at
all). Both schemas exhibit the same artifact, so the detector lives here rather
than in either consumer.

Root cause investigation (2026-09-18, both the live Jetson evaluation runs and
the four raw training sessions behind Dataset 8 -- host_software/data/01_bronze/
session_20260810_{104132,110239,112047,114330}/telemetry.csv): a small fraction
of touch_x/touch_y readings from the resistive touch-plate sensor
(firmware/stm32_ml_control_and_vision/BallBalancingBot/TouchProbe.cpp) are
isolated, single-frame, simultaneous positive-going jumps in both axes that
revert to trend on the very next sample. Confirmed NOT reliably caught by two
weaker approaches:
  - The firmware's own touch_valid flag: most flagged spikes in the Jetson
    telemetry are reported touch_valid=1 (a 2-of-3-samples-above-a-low-pressure-
    threshold majority vote that is not a reliable ground-truth-plausibility
    check).
  - A simple physical-bounds check alone: many spikes (60-90mm) land inside the
    platform's actual physical extent and would pass a bounds-only filter.
Hypothesis (not confirmed at the circuit level -- no raw ADC counts are logged,
only the final mm value): TouchProbe.cpp's touch/drive pins are dynamically
reconfigured every scan (see its own comment: "getPoint() leaves the shared
pins configured for reading; restore them"), a plausible source of a transient
glitch during that role switch. Flagged to the Electrical Engineer / firmware
owner as a real, reproducible artifact worth investigating in hardware -- not
resolved here.

Detection strategy (revised 2026-09-18, v2): flag row i if it deviates from the
MEDIAN of up to 4 surrounding DISTINCT-VALUED rows (2 before, 2 after, after
first collapsing consecutive exact-duplicate rows to one representative) by
more than jump_mm. This fixes two real gaps found in a first design (v1, a
strict "compare only to the single immediate neighbor on each side, which must
themselves agree" check) by inspecting run3/run4/run5's still-visible spikes
after v1 filtering:
  1. v1 never flagged the FIRST or LAST row of a run (no i-1/i+1 to compare
     against). In practice, several runs' first ~50 rows (~2s at 25Hz) hold a
     STALE value from before the current recording started (confirmed directly:
     ground_truth_jetson_20260915_151627.csv holds touch_x=-88.35/touch_y=-29.4,
     touch_valid=0, for rows 0-49, then jumps to the real trajectory at row 50)
     -- not the same single-frame-glitch mechanism, but a real, distinct
     "pre-recording stale state" artifact.
  2. v1's "immediate neighbors must agree within 15mm" condition can fail
     during genuinely fast ball oscillation (confirmed in run3 around a mid-run
     spike near a rapid direction change): if the two flanking samples
     themselves differ by more than the agreement threshold due to real motion,
     not corruption, v1 never flags the point between them even though it is a
     clear outlier against the local trend.

The DISTINCT-VALUED part is not optional polish -- an early v2 attempt that
built the 4-neighbor window from raw consecutive rows (without deduplication)
regressed badly (run1 alone went from 5 to 115 flagged frames, nearly all
false positives on ordinary movement): TouchProbe.cpp holds the exact last
value on every touch_valid=0 row, so raw consecutive rows routinely contain
runs of identical held values interleaved with genuinely-updated ones. A
window mixing "stale, repeated" and "fresh, moving" values does not represent
a clean local trend, and its median can land meaningfully off-trend even for
an entirely legitimate reading. Collapsing consecutive exact duplicates to one
representative value before building the window (see _distinct_value_indices())
fixes this directly, and as a side effect this same mechanism also catches the
leading-stale-segment case (item 1 above) without needing a separate touch_valid-
based rule: 50 identical leading rows collapse to a single distinct value,
which is then compared against the real trajectory's first few distinct values
and correctly flagged as far outside them. A detected flag is propagated back
across every row in its original duplicate run (see flag_isolated_spikes()).

v2 also resolves the "two consecutive corrupted frames" gap v1's docstring
flagged as a known limitation: with a 4-distinct-neighbor median, a second bad
value among the 4 no longer prevents correct detection of either bad frame
(median tolerates one contaminated value out of four).

A design using a rolling-median/MAD (Hampel) *window statistic* over raw rows
was tried before v1 and rejected for the same held-duplicate-value reason
described above (a near-zero local MAD flagged ordinary small real movements).
v2's fixed absolute threshold over de-duplicated neighbors avoids both known
degeneracies.
"""

import numpy as np
import pandas as pd

# firmware/stm32_ml_control_and_vision/BallBalancingBot/TouchProbe.cpp's own
# calibration constants (SCREEN_WIDTH_MM=187.5, SCREEN_HEIGHT_MM=141.0) -- the
# actual extent the logged touch_x/touch_y values are computed against. Note
# this differs by 1mm on the Y axis from CLAUDE.md's nominal platform spec
# (142.0mm) -- a separate, pre-existing firmware/spec discrepancy, not
# something this filter corrects; using the firmware's own effective value
# since that's what the logged data was actually generated against.
TOUCH_PLATE_X_BOUND_MM = 187.5 / 2.0
TOUCH_PLATE_Y_BOUND_MM = 141.0 / 2.0

# Chosen from direct inspection of ~90+ manually confirmed spikes across the 5
# Track 1 evaluation runs and the 4 raw Dataset 8 training sessions
# (2026-09-18): every genuine spike departs from its local neighborhood by
# >=40mm; the largest genuine frame-to-frame ball movement observed during
# normal (non-spike) oscillation, including fast direction changes, was well
# under 20mm. SPIKE_JUMP_MM sits comfortably above real dynamics and below
# every observed spike magnitude.
SPIKE_JUMP_MM = 30.0

# Neighbors on each side used to build the robust local-trend estimate (median
# of up to 2*HALF_WINDOW values). HALF_WINDOW=2 tolerates one corrupted value
# among the up-to-4 neighbors without the median itself being pulled off trend.
HALF_WINDOW = 2

def _distinct_value_run_starts(x, y):
    """Returns the index of the first row of each run of consecutive exact
    duplicate (x, y) values -- i.e. one representative index per distinct
    value, in original temporal order. TouchProbe.cpp emits an exact repeat of
    the last known position on every touch_valid=0 row, so this is how the
    "fresh" reading sequence is recovered without needing touch_valid itself
    (works identically on the older training-session schema, which has no
    touch_valid column at all)."""
    n = len(x)
    if n == 0:
        return np.array([], dtype=int)
    is_new = np.ones(n, dtype=bool)
    is_new[1:] = (x[1:] != x[:-1]) | (y[1:] != y[:-1])
    return np.flatnonzero(is_new)


def flag_isolated_spikes(touch_x, touch_y, jump_mm=SPIKE_JUMP_MM, half_window=HALF_WINDOW):
    """Flags row i where dist(point i, median of its up-to-2*half_window
    surrounding DISTINCT-valued neighbors) > jump_mm -- see module docstring
    for why deduplication matters. Neighbor windows shrink naturally at the
    start/end of the run (as few as half_window neighbors there) rather than
    excluding boundary rows outright, so a stale value at the very start of a
    recording can still be flagged. Requires at least 2 distinct neighbors to
    make a call. A flagged distinct value is propagated to every original row
    in its duplicate run. Returns a boolean numpy array aligned to touch_x's
    original row order (this is a temporal check, so the input must already be
    in acquisition order)."""
    x = np.asarray(touch_x, dtype=float)
    y = np.asarray(touch_y, dtype=float)
    n = len(x)
    flagged = np.zeros(n, dtype=bool)
    if n == 0:
        return flagged

    run_starts = _distinct_value_run_starts(x, y)
    m = len(run_starts)
    dx, dy = x[run_starts], y[run_starts]
    run_flagged = np.zeros(m, dtype=bool)

    for k in range(m):
        lo = max(0, k - half_window)
        hi = min(m, k + half_window + 1)
        neighbor_k = [j for j in range(lo, hi) if j != k]
        if len(neighbor_k) < 2:
            continue
        med_x = np.median(dx[neighbor_k])
        med_y = np.median(dy[neighbor_k])
        if np.hypot(dx[k] - med_x, dy[k] - med_y) > jump_mm:
            run_flagged[k] = True

    # Propagate each flagged distinct value across every original row in its
    # duplicate run: [run_starts[k], run_starts[k+1]) (or to n for the last run).
    run_ends = np.append(run_starts[1:], n)
    for k in range(m):
        if run_flagged[k]:
            flagged[run_starts[k]:run_ends[k]] = True

    return flagged


def flag_leading_invalid_segment(touch_valid):
    """Flags every row from the start of the run up to (not including) the
    first touch_valid==1 row. Ground truth is undefined before the touch
    sensor's first confirmed contact -- whatever value the firmware reports
    during that window (commonly a stale position held over from before this
    recording started, e.g. -88.35mm/-29.4mm for 50 rows at the start of
    ground_truth_jetson_20260915_151627.csv) is not a real ball position.
    Returns an all-False array if touch_valid is None or never 1 (nothing to
    anchor the trim to -- left unflagged rather than guessed at)."""
    if touch_valid is None:
        return None
    valid = np.asarray(touch_valid)
    n = len(valid)
    flagged = np.zeros(n, dtype=bool)
    valid_idx = np.flatnonzero(valid == 1)
    if valid_idx.size == 0:
        return flagged
    first_valid = valid_idx[0]
    flagged[:first_valid] = True
    return flagged


def flag_touch_position_outliers(
    touch_x,
    touch_y,
    touch_valid=None,
    x_bound_mm=TOUCH_PLATE_X_BOUND_MM,
    y_bound_mm=TOUCH_PLATE_Y_BOUND_MM,
    jump_mm=SPIKE_JUMP_MM,
    half_window=HALF_WINDOW,
):
    """Combines the physical-bounds hard-reject, the windowed-median isolated-
    spike check, and (when touch_valid is supplied) the leading-invalid-segment
    trim. Returns a boolean Series (True = corrupted / should be excluded),
    aligned to touch_x's index. touch_x/touch_y (and touch_valid, if given)
    should be a single run's readings in original row order (the temporal
    checks require this)."""
    touch_x = pd.Series(touch_x, dtype=float)
    touch_y = pd.Series(touch_y, dtype=float)

    out_of_bounds = (touch_x.abs() > x_bound_mm) | (touch_y.abs() > y_bound_mm)
    isolated_spike = pd.Series(
        flag_isolated_spikes(touch_x.to_numpy(), touch_y.to_numpy(), jump_mm=jump_mm, half_window=half_window),
        index=touch_x.index,
    )
    result = out_of_bounds | isolated_spike

    leading_invalid = flag_leading_invalid_segment(touch_valid)
    if leading_invalid is not None:
        result = result | pd.Series(leading_invalid, index=touch_x.index)

    return result
