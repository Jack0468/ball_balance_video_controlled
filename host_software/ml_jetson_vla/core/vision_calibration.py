"""
Per-session vision calibration correction for the small-class expert pipeline's
CNN ball-position estimate.

Motivation, from a real overnight investigation (2026-09-23/24) into why Track 1's
Steady-State Error is well above the <3mm target: vision's own accuracy against
touch-sensor ground truth (~5-6mm mean error, pooled over 10 real Track 4 sessions,
40,713 frames) is a real, non-trivial contributor -- the control loop closes on
vision, not touch (touch is a sensor-only evaluation reference, per
BallBalancingBot.ino's own header comment), so a vision calibration bias shows up
almost directly as Steady-State Error, not just as evaluation noise.

Root cause, established by ruling out alternatives one at a time against real data,
not assumed:
  - NOT serial/integration staleness (MCU-echoed vision matched host-sent vision to
    0.01mm; touch/motor telemetry matched vision `seq` 1:1 in 95% of frames).
  - NOT a touch-sensor noise floor (settled-window touch position shows lag-1
    autocorrelation 0.64-0.93 -- real correlated motion, not white sensor noise).
  - NOT per-frame ArUco/homography estimation noise (an ml-vision subagent tested
    anchoring the homography from an average of 150 early frames per session
    instead of recomputing it fresh every frame -- this made error WORSE in 9/10
    sessions, and the position-dependent error's R^2 did not drop, which it should
    have if per-frame noise were the cause).
  - NOT radial lens distortion (correlation between error and distance from
    platform center: r=-0.0008, flat across radial octiles).
  - NOT a fixed touch-sensor calibration nonlinearity (the position-dependent
    [quadratic] error component's coefficient of variation across sessions is
    0.96-4.81 -- a fixed hardware property of one sensor would be stable, this
    isn't).
  - Best-supported explanation: each recording session has its own slightly
    different physical camera/platform geometry (not fixable by better estimation
    of the SAME session, since the geometry itself differs each time recording
    starts) -- producing a session-specific affine (position-dependent) error
    pattern, stacked on a separately-stable global Y-axis bias consistent with the
    coordinate-frame mismatch firmware/stm32_ml_control_and_vision/BallBalancingBot/
    TouchProbe.cpp's own "FRAME MATCH" comment already flags and which was never
    resolved (its own 5-point calibration procedure was never run).

What this module provides: the fit/apply/blend math for a per-session affine
correction, validated (see validate_vision_calibration.py in ../deployment/) via
proper held-out testing (calibration and test frames disjoint, same session) on
real recorded data:
  - No correction:                        5.64mm mean vision error (pooled)
  - Per-session affine correction:        4.41mm (21.8% reduction)
  - Global (all-sessions-pooled) correction only: 4.96mm (12.1% reduction --
    real, but captures only the stable component, not the session-specific one)
  - Session correction blended 75/25 with the global correction:
    4.36mm (22.7% reduction) -- the best validated result, and the recommended
    default (see blend_corrections()'s alpha default).
A naive "use whatever the ball visits in the first few seconds" calibration
window was tried FIRST and made things 2-4x WORSE (extrapolation far outside the
narrow region that incidental early motion actually covers) -- calibration
samples must have deliberate, spread-out spatial coverage (see
generate_calibration_grid()), not just be "early."

======================================================================================
STATUS (2026-09-24): NOT INTEGRATED, NOT HARDWARE-TESTED. THIS IS A TODO, NOT A FIX.
======================================================================================
Every number above comes from OFFLINE re-analysis of previously-recorded Track 4
session video + telemetry (re-running the production ArUco+CNN pipeline against
saved frames). No live hardware run has ever exercised this module. Concretely
outstanding before this could be trusted on real hardware:

  1. REGRESSOR VARIABLE MISMATCH, NOT YET RE-VALIDATED. The validation above fit
     err = f(touch_x, touch_y) -- the TRUE position -- because that's the only
     sound way to characterize what the error actually depends on. But at
     inference time, after a calibration phase ends, touch is not fed into this
     loop (see the Coordinate Contract in CLAUDE.md and BallBalancingBot.ino's
     "touchscreen is a SENSOR ONLY" comment) -- only the CNN's own raw_x/raw_y is
     available. This module's fit_affine_correction() is therefore written to
     regress against whatever `positions` array the caller passes in, and the
     intended live-deployment call passes (raw vision x, raw vision y) as that
     array, not (touch_x, touch_y). Since raw_x/raw_y is close to touch_x/touch_y
     by construction (the very quantity being corrected is small relative to
     platform scale), this should behave similarly -- but that is a reasoned
     expectation, not a re-run test. Re-validate with vision position as the
     regressor specifically before trusting the exact 4.36mm number.
  2. NO WAY TO ACTUALLY DRIVE THE BALL TO A CALIBRATION GRID POINT YET.
     TargetStateMachine (src/state_machine.py) only supports "center", "hold", or
     a live marker-color average as a target -- there is no "go to this exact
     (x_mm, y_mm)" command. CalibrationSampleCollector below assumes samples
     arrive somehow; it does not drive anything. Needs a new target-setting
     capability (or a manual/scripted physical placement procedure) before a real
     calibration phase can run unattended.
  3. NOT WIRED INTO run_jetson_standalone.py. JetsonExpertPolicy.act() computes
     raw_x, raw_y via px_to_touch_mm() and passes them directly to
     PredictionGate.filter() -- see that file's comment at the call site for
     exactly where apply_correction()'s output would need to be inserted instead.
     Not done here on purpose: this changes CLAUDE.md's LOCKED "ArUco homography ->
     warped -> pixel x scale = mm (no MLP)" coordinate mapping decision (this is a
     simple affine fit, not an MLP, but it IS a new parametrized layer on top of
     the existing analytic mapping) and needs explicit user sign-off first.
  4. NO STANDING GLOBAL-CORRECTION WORKFLOW. fit_global_correction() computes a
     global correction from a set of past sessions' samples, but this module does
     not define how/when that gets re-fit and persisted over time (e.g. after
     every N new sessions) -- that's a deployment/ops decision, not made here.

See host_software/ml_jetson_vla/docs/VISION_CALIBRATION_PROPOSAL.md for the full
writeup, validation methodology, and the specific hardware experiments recommended
to close gaps 1-2 above (self_test() below only checks structural correctness --
shapes, round-tripping, blend arithmetic -- exactly the same limitation
control_net.py's own "NOT YET HARDWARE-VALIDATED" banner describes for that file).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class AffineCorrection:
    """err_x/err_y modeled as affine functions of a 2D position:
        err_x = ax[0]*px + ax[1]*py + ax[2]
        err_y = ay[0]*px + ay[1]*py + ay[2]
    `px, py` is whatever position array the correction was fit against -- see the
    module docstring's "REGRESSOR VARIABLE MISMATCH" warning. This dataclass does
    not know or care which one was used; the caller must be consistent between
    fit_affine_correction() and apply_correction()."""

    ax: Tuple[float, float, float]
    ay: Tuple[float, float, float]

    def to_json_dict(self) -> dict:
        return {"ax": list(self.ax), "ay": list(self.ay)}

    @staticmethod
    def from_json_dict(d: dict) -> "AffineCorrection":
        ax = d["ax"]
        ay = d["ay"]
        return AffineCorrection(ax=(ax[0], ax[1], ax[2]), ay=(ay[0], ay[1], ay[2]))

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_json_dict(), f, indent=2)

    @staticmethod
    def load(path: str) -> "AffineCorrection":
        with open(path) as f:
            return AffineCorrection.from_json_dict(json.load(f))


# A no-op correction (all coefficients zero) -- useful as a safe default before any
# calibration has been fit, or as the "global" side of blend_corrections() the very
# first time a global correction doesn't exist yet.
IDENTITY_CORRECTION = AffineCorrection(ax=(0.0, 0.0, 0.0), ay=(0.0, 0.0, 0.0))

# Recommended default from the 2026-09-24 validation (see module docstring) --
# 75% weight on a session's own calibration, 25% on the standing global average.
# Swept 0.0-1.0 in steps of 0.1 against real held-out data; 0.7-0.8 was the flat
# optimum (4.357-4.361mm), 0.75 chosen as the midpoint, not because 0.75 itself
# was separately tested.
DEFAULT_BLEND_ALPHA = 0.75


def fit_affine_correction(
    positions: Sequence[Tuple[float, float]],
    errors: Sequence[Tuple[float, float]],
) -> AffineCorrection:
    """Ordinary least squares fit of err = f(position), one 3-parameter fit per
    axis (np.linalg.lstsq, not a regularized/ridge fit -- see blend_corrections()
    for how this module handles the small-sample-size overfitting risk instead of
    regularizing the fit itself).

    positions, errors: parallel sequences of (x, y) tuples, same length, same
    units (mm) and same coordinate convention (0,0 = platform center) as
    touch_x_mm/touch_y_mm elsewhere in this project. errors[i] must be the error
    OBSERVED AT positions[i] -- i.e. errors[i] = (vision_x[i] - touch_x[i],
    vision_y[i] - touch_y[i]) if positions[i] is a vision reading, or the
    equivalent if positions[i] is a touch reading (see module docstring for which
    one the live deployment path is intended to use).

    Raises ValueError if fewer than 4 samples are given (need >=4 to fit 3
    parameters without a degenerate/underdetermined system; the validated
    deployment default uses ~107, this floor is a correctness guard, not a
    recommendation for how few samples to actually use)."""
    if len(positions) != len(errors):
        raise ValueError(f"positions and errors must be the same length, got {len(positions)} and {len(errors)}")
    if len(positions) < 4:
        raise ValueError(f"need at least 4 samples to fit a 3-parameter affine model per axis, got {len(positions)}")

    pos = np.asarray(positions, dtype=np.float64)
    err = np.asarray(errors, dtype=np.float64)
    design = np.column_stack([pos[:, 0], pos[:, 1], np.ones(len(pos))])
    ax_fit, *_ = np.linalg.lstsq(design, err[:, 0], rcond=None)
    ay_fit, *_ = np.linalg.lstsq(design, err[:, 1], rcond=None)
    return AffineCorrection(
        ax=(float(ax_fit[0]), float(ax_fit[1]), float(ax_fit[2])),
        ay=(float(ay_fit[0]), float(ay_fit[1]), float(ay_fit[2])),
    )


def apply_correction(raw_x: float, raw_y: float, correction: AffineCorrection) -> Tuple[float, float]:
    """Subtract the correction's predicted error from a raw position estimate.
    `raw_x, raw_y` must be in the same units/convention the correction was fit
    against (see AffineCorrection's docstring)."""
    ax1, ax2, ax3 = correction.ax
    ay1, ay2, ay3 = correction.ay
    predicted_err_x = ax1 * raw_x + ax2 * raw_y + ax3
    predicted_err_y = ay1 * raw_x + ay2 * raw_y + ay3
    return raw_x - predicted_err_x, raw_y - predicted_err_y


def blend_corrections(
    session_correction: AffineCorrection,
    global_correction: AffineCorrection,
    alpha: float = DEFAULT_BLEND_ALPHA,
) -> AffineCorrection:
    """Linear shrinkage blend of a session-specific fit toward a standing global
    average, coefficient-by-coefficient: alpha=1.0 is pure per-session (lowest
    bias, highest variance -- vulnerable to a noisy small calibration sample),
    alpha=0.0 is pure global (lowest variance, but blind to genuine session-
    specific error -- validated 12.1% reduction vs. per-session's 21.8%).

    Why blending beats picking one or the other: validated on real held-out data
    that some individual sessions' own ~107-point calibration fit is noisy enough
    to underperform the global average outright (e.g. one session: per-session
    5.82mm vs. global 4.59mm on the same held-out test set) -- shrinking toward
    the global average trades a little bit of session-specificity for protection
    against exactly that failure mode, and empirically wins on average (4.36mm
    pooled at alpha=0.75, beating both pure alternatives)."""
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0.0, 1.0], got {alpha}")
    ax = tuple(alpha * s + (1.0 - alpha) * g for s, g in zip(session_correction.ax, global_correction.ax))
    ay = tuple(alpha * s + (1.0 - alpha) * g for s, g in zip(session_correction.ay, global_correction.ay))
    return AffineCorrection(ax=(ax[0], ax[1], ax[2]), ay=(ay[0], ay[1], ay[2]))


def fit_global_correction(
    session_samples: Iterable[Tuple[Sequence[Tuple[float, float]], Sequence[Tuple[float, float]]]],
) -> AffineCorrection:
    """Pool multiple sessions' (positions, errors) sample sets into ONE fit --
    the "global" side of blend_corrections(). Deliberately does not weight
    sessions by anything other than raw sample count (a session that happened to
    collect more calibration samples contributes proportionally more) -- no
    evidence was gathered on whether a smarter weighting (e.g. by session
    recency) would help, so this stays simple until there's a reason not to.

    Does NOT persist anything to disk or decide when it should be re-run against
    a growing history of sessions -- that workflow is not defined by this module
    (see module docstring, gap 4)."""
    all_positions: List[Tuple[float, float]] = []
    all_errors: List[Tuple[float, float]] = []
    for positions, errors in session_samples:
        all_positions.extend(positions)
        all_errors.extend(errors)
    return fit_affine_correction(all_positions, all_errors)


def generate_calibration_grid(
    platform_width_mm: float,
    platform_height_mm: float,
    grid_size: int = 3,
    margin_frac: float = 0.15,
) -> List[Tuple[float, float]]:
    """Deliberate, spread-out calibration target positions -- NOT a substitute
    for "whatever the ball visits early in a session" (validated to be actively
    harmful, see module docstring: naive early-window calibration made error
    2-4x WORSE due to extrapolating a linear fit far outside the narrow region it
    was fit from).

    Returns grid_size*grid_size (x_mm, y_mm) points in the centered coordinate
    convention (0,0 = platform center) matching touch_x_mm/touch_y_mm elsewhere
    in this project, spanning the platform inset by margin_frac of each axis's
    half-extent (so grid points aren't at the literal physical edge, where the
    ball is more likely to be lost/occluded during a real calibration pass).
    grid_size=3 (9 points) matches what was validated -- 2026-09-24's held-out
    test used up to 15 samples/cell (~107 total after the settled-sample filter
    in CalibrationSampleCollector); other grid sizes are untested."""
    if grid_size < 2:
        raise ValueError(f"grid_size must be >= 2 to bound both axes, got {grid_size}")
    half_w = (platform_width_mm / 2.0) * (1.0 - margin_frac)
    half_h = (platform_height_mm / 2.0) * (1.0 - margin_frac)
    xs = np.linspace(-half_w, half_w, grid_size)
    ys = np.linspace(-half_h, half_h, grid_size)
    return [(float(x), float(y)) for y in ys for x in xs]


class CalibrationSampleCollector:
    """Accumulates (raw_vision_xy, touch_xy) sample pairs for ONE calibration
    grid point, and decides when enough consistent samples have been collected --
    mirrors PredictionGate's seed-window pattern (N consistent frames within a
    tolerance, not a fixed timeout) so a point that settles quickly doesn't waste
    calibration time and a slow one doesn't get under-sampled.

    Does NOT drive the ball to target_xy -- see module docstring gap 2.
    Something else (not yet built) has to get the ball there; this class only
    watches the touch-sensor stream to decide once it plausibly has and records
    matching vision/touch pairs while that holds."""

    def __init__(
        self,
        target_xy: Tuple[float, float],
        required_samples: int = 12,
        consistency_tolerance_mm: float = 8.0,
    ) -> None:
        self.target_xy = target_xy
        self.required_samples = required_samples
        self.consistency_tolerance_mm = consistency_tolerance_mm
        self._vision_samples: List[Tuple[float, float]] = []
        self._touch_samples: List[Tuple[float, float]] = []

    def add_sample(self, raw_vision_xy: Tuple[float, float], touch_xy: Tuple[float, float]) -> None:
        self._vision_samples.append(raw_vision_xy)
        self._touch_samples.append(touch_xy)

    @property
    def is_settled(self) -> bool:
        """True once the most recent `required_samples` touch readings agree
        within consistency_tolerance_mm of each other (max-min spread on each
        axis, combined via Euclidean norm) -- i.e. the ball has actually arrived
        and is holding still at this grid point, not still in transit toward
        it."""
        if len(self._touch_samples) < self.required_samples:
            return False
        recent = np.asarray(self._touch_samples[-self.required_samples:], dtype=np.float64)
        spread = float(np.linalg.norm(recent.max(axis=0) - recent.min(axis=0)))
        return spread <= self.consistency_tolerance_mm

    def settled_samples(self) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """(vision positions, touch positions) from the settled tail only --
        drops whatever transient arrival period preceded settling at this point.
        Raises RuntimeError if called before is_settled is True; the caller must
        check first, this does not silently return a partial/unsettled window."""
        if not self.is_settled:
            raise RuntimeError("settled_samples() called before is_settled -- caller must check is_settled first")
        return (
            self._vision_samples[-self.required_samples:],
            self._touch_samples[-self.required_samples:],
        )


def self_test() -> None:
    """Structural correctness only (shapes, round-tripping, blend arithmetic on
    synthetic data with a known ground-truth correction) -- NOT a substitute for
    the real held-out validation against recorded hardware data (see
    ../deployment/validate_vision_calibration.py), same limitation
    control_net.py's own self_test() has and says so for the same reason: no live
    hardware access from this environment to validate against."""
    # Recover a known, synthetic affine error model from noiseless synthetic data.
    rng = np.random.default_rng(0)
    true_ax = (0.05, -0.02, 3.0)
    true_ay = (-0.01, 0.03, -1.5)
    positions = [(float(x), float(y)) for x, y in rng.uniform(-80, 80, size=(50, 2))]
    errors = [
        (true_ax[0] * x + true_ax[1] * y + true_ax[2], true_ay[0] * x + true_ay[1] * y + true_ay[2])
        for x, y in positions
    ]
    fitted = fit_affine_correction(positions, errors)
    assert np.allclose(fitted.ax, true_ax, atol=1e-6), f"ax fit mismatch: {fitted.ax} vs {true_ax}"
    assert np.allclose(fitted.ay, true_ay, atol=1e-6), f"ay fit mismatch: {fitted.ay} vs {true_ay}"

    # apply_correction() should exactly invert the known synthetic error WHEN
    # EVALUATED AT THE SAME POSITION THE MODEL WAS FIT AGAINST (true_x, true_y
    # here, matching `positions` above) -- this checks the arithmetic is correct
    # in isolation. It deliberately does NOT apply the correction at the raw
    # (error-added) position instead: doing so would conflate this structural
    # check with the module's own documented "regressor variable mismatch" gap
    # (module docstring, point 1) -- evaluating at a position that itself
    # already contains the error being corrected is an approximation by
    # construction, not exactly invertible, and asserting exact recovery there
    # would be asserting something mathematically false, not verifying the code.
    test_x, test_y = 40.0, -20.0
    err_x = true_ax[0] * test_x + true_ax[1] * test_y + true_ax[2]
    err_y = true_ay[0] * test_x + true_ay[1] * test_y + true_ay[2]
    raw_x, raw_y = test_x + err_x, test_y + err_y
    corrected_x, corrected_y = apply_correction(test_x, test_y, fitted)
    assert abs(corrected_x - (test_x - err_x)) < 1e-6 and abs(corrected_y - (test_y - err_y)) < 1e-6, (
        f"apply_correction arithmetic is wrong: got ({corrected_x}, {corrected_y}), "
        f"expected ({test_x - err_x}, {test_y - err_y})"
    )
    # Sanity-check the approximation gap itself is small but nonzero here (raw_x
    # was displaced from test_x by err_x, which is itself only a few mm at this
    # test point) -- confirms the two evaluation points genuinely differ, i.e.
    # this test isn't accidentally checking the same thing twice.
    assert abs(raw_x - test_x) > 1e-6 or abs(raw_y - test_y) > 1e-6

    # blend_corrections() at alpha=1.0 must equal the session correction exactly,
    # at alpha=0.0 must equal the global correction exactly.
    session = AffineCorrection(ax=(1.0, 2.0, 3.0), ay=(4.0, 5.0, 6.0))
    global_ = AffineCorrection(ax=(0.1, 0.2, 0.3), ay=(0.4, 0.5, 0.6))
    assert blend_corrections(session, global_, alpha=1.0) == session
    assert blend_corrections(session, global_, alpha=0.0) == global_
    midpoint = blend_corrections(session, global_, alpha=0.5)
    assert np.allclose(midpoint.ax, (0.55, 1.1, 1.65)), midpoint.ax

    # JSON round-trip.
    dumped = fitted.to_json_dict()
    reloaded = AffineCorrection.from_json_dict(dumped)
    assert reloaded == fitted

    # generate_calibration_grid() returns the right count, spans the expected
    # inset range, and is centered.
    grid = generate_calibration_grid(187.5, 142.0, grid_size=3, margin_frac=0.15)
    assert len(grid) == 9, f"expected 9 grid points, got {len(grid)}"
    xs = [p[0] for p in grid]
    expected_half_w = (187.5 / 2.0) * 0.85
    assert abs(max(xs) - expected_half_w) < 1e-6 and abs(min(xs) + expected_half_w) < 1e-6, grid

    # CalibrationSampleCollector: not settled below the sample threshold, becomes
    # settled once enough consistent touch readings arrive, and correctly
    # rejects a query for samples before that.
    collector = CalibrationSampleCollector(target_xy=(0.0, 0.0), required_samples=5, consistency_tolerance_mm=8.0)
    for i in range(4):
        collector.add_sample((float(i), 0.0), (float(i), 0.0))
    assert not collector.is_settled, "should not be settled with only 4/5 required samples"
    try:
        collector.settled_samples()
        raise AssertionError("settled_samples() should have raised before is_settled")
    except RuntimeError:
        pass
    collector.add_sample((0.0, 0.0), (0.0, 0.0))  # 5th sample, all touch readings within tolerance
    assert collector.is_settled, "should be settled with 5 consistent samples"
    vis, touch = collector.settled_samples()
    assert len(vis) == 5 and len(touch) == 5

    print("vision_calibration.py self_test(): all checks passed (structural only -- NOT hardware-validated)")


if __name__ == "__main__":
    self_test()
