# Per-Session Vision Calibration — Proposal

**Status: proposal + validated offline analysis, NOT integrated, NOT hardware-tested.
This is a TODO, not a shipped fix.** Everything below comes from re-analyzing
previously-recorded video/telemetry; no live Jetson/STM32 run has ever exercised this
code. See "What's needed before this touches hardware" at the end before anyone wires
this into `run_jetson_standalone.py`.

## Why this exists

Investigating why Track 1's Steady-State Error sits well above the project's <3mm
target (2026-09-23/24 overnight investigation), it became clear the control loop
closes on the CNN's own vision estimate, never on touch-sensor ground truth (see
`BallBalancingBot.ino`'s own header: "The touchscreen is a SENSOR ONLY. It never feeds
the controller."). That means any systematic bias in vision's own position estimate
shows up almost directly as Steady-State Error, not just as evaluation noise — fixing
vision's accuracy against touch ground truth is not a side project, it's a direct lever
on the metric the target is defined against.

## Root cause, established by elimination against real data

Each hypothesis below was tested and ruled out (or confirmed) against real recorded
data — see `docs/PROJECT_LOGBOOK.md`'s 2026-09-23/24 entries for the full chain,
summarized here:

| Hypothesis | Test | Result |
|---|---|---|
| Serial/integration staleness | Compared MCU-echoed vision position to host-sent; checked touch-telemetry-row-per-vision-`seq` ratio | Ruled out — matched to 0.01mm; 95% of frames 1:1 |
| Touch-sensor noise floor | Lag-1 autocorrelation of settled-window touch position | Ruled out — 0.64-0.93, the signature of real correlated motion, not white sensor noise |
| Per-frame ArUco/homography estimation noise | `ml-vision` subagent: anchored homography from 150 averaged early frames vs. fresh per-frame, re-ran full pipeline over all 10 real Track 4 sessions (40,713 frames) | Ruled out — made error WORSE in 9/10 sessions (pooled 6.40mm → 6.86mm); position-dependent error's R² did not drop, which averaging away noise should have caused if noise were the source |
| Radial lens distortion | Correlation of error magnitude vs. distance from platform center | Ruled out — r = -0.0008, flat across radial octiles |
| Fixed touch-sensor calibration nonlinearity | Coefficient of variation of the quadratic (position-dependent) error terms, across sessions | Ruled out — CoV 0.96-4.81 (often bigger than the mean, several terms flip sign); a fixed hardware property of one sensor would be stable across sessions using it |
| **Session-specific physical geometry** (camera/platform position/angle differs slightly each time recording starts) | Affine fit of err = f(position), per session; check whether the fitted slope/intercept are stable across sessions on the same day | **Best-supported explanation** — R² up to 0.22 (real, position-dependent structure), coefficients vary substantially session-to-session even within the same ~90-minute window on the same day |

This is layered on top of a **separately stable, global positive Y-axis bias**
(+1.57 to +3.04mm, consistent across all 10 sessions) — consistent with the
coordinate-frame mismatch `TouchProbe.cpp`'s own "FRAME MATCH" comment already flags
and whose documented 5-point calibration procedure was never run.

## The technique

A per-session affine correction: `err_x = a1·x + a2·y + a3`, `err_y = b1·x + b2·y +
b3`, fit from a deliberately spread-out calibration sample (a 3×3 grid across the
platform, not "whatever the ball visits early" — see below for why that distinction
matters), applied to the CNN's raw position estimate before it reaches
`PredictionGate`.

Implementation: `host_software/ml_jetson_vla/core/vision_calibration.py`
(`AffineCorrection`, `fit_affine_correction`, `apply_correction`,
`blend_corrections`, `fit_global_correction`, `generate_calibration_grid`,
`CalibrationSampleCollector`). Structural correctness verified via that module's
`self_test()` — passes, but per its own docstring this is **not** a substitute for
real hardware validation.

### A naive version was tried first, and it made things worse

The first design used "whatever the ball naturally visits in the first 10-30% of a
session" as the calibration window. Tested on all 10 real sessions: **error got 2-4x
WORSE** (up to 33mm mean, vs. ~5-9mm raw) on the held-out remainder of each session.
Root cause, directly confirmed by comparing each session's early-window coordinate
range against its full range: natural early motion often never reaches the platform
edges, so the fitted linear model was being extrapolated far outside the region it was
fit from. **Calibration data must have deliberate, representative spatial coverage** —
this is why `generate_calibration_grid()` exists rather than just recording early
frames.

### Validated results

Two independent validation runs, both proper held-out tests (calibration and test
frames disjoint, within the same session):

| Method | Run 1 (ad hoc `pd.cut` binning, ~107 calib pts/session) | Run 2 (`validate_vision_calibration.py`, nearest-neighbor to grid points, ~135 calib pts/session) |
|---|---|---|
| No correction | 5.64mm | 5.63mm |
| Per-session affine correction | 4.41mm (−21.8%) | 4.51mm (−19.9%) |
| Global correction (all sessions pooled, one fit) | 4.96mm (−12.1%) | 5.21mm (−7.5%) |
| Session blended 75/25 with global | **4.36mm (−22.7%)** | 4.57mm (−18.8%, *worse* than pure per-session) |

**Read this table carefully — it is not fully consistent, and that's the honest
result, not a mistake papered over:**

- **Robust across both runs**: per-session affine correction gives a real ~20-22%
  reduction in vision error. This is the core, trustworthy finding.
- **Robust across both runs**: pure global correction helps some, less than
  per-session — it only captures the stable (Y-axis) component, not the
  session-specific one.
- **NOT robust**: whether blending toward the global correction beats pure
  per-session. It won in Run 1 (4.36 < 4.41) and lost in Run 2 (4.57 > 4.51). The
  fixed `alpha=0.75` default in `vision_calibration.py` was chosen from Run 1's
  sweep alone; Run 2 was not swept over alpha. **Don't trust the blending
  recommendation as-is** — either re-sweep alpha against a larger, independent
  sample before relying on it, or default to pure per-session (`alpha=1.0`) until
  that's done, since pure per-session is the more consistently validated choice.

### Bottom line against the <3mm target

Best validated result (~4.4-4.5mm) is still above 3mm. This is a real, worthwhile
improvement (roughly halves the gap between the current ~9-17mm Steady-State Error
readings and target) but **not sufficient alone** — reaching <3mm will need a second
lever, most likely control-loop/settling dynamics (Kalman R/Q, `PredictionGate`
parameters), not yet investigated as deeply as the vision side.

## Two deployment variants — an open design decision, not resolved here

Every validated number above uses touch position **at both fit time and apply time**
(an "oracle" evaluation — see `validate_vision_calibration.py`'s own docstring). This
matters because it assumes continuous access to touch-sensor ground truth, not just
during an initial calibration window, which is a genuine departure from this
project's stated design principle that touch is a sensor-only, evaluation-independent
reference (`BallBalancingBot.ino`'s comment). Two ways to resolve this, neither built
or tested:

1. **Calibrate once at session start, apply vision-only afterward.** Fit the
   correction while touch is available (a deliberate calibration phase), then for the
   rest of the session evaluate `apply_correction()` using the CNN's own raw position
   only — touch no longer read for this purpose. Architecturally clean, matches the
   stated sensor-independence principle, but **completely untested** — the validated
   numbers above do not describe this variant's real accuracy, since raw vision
   position ≈ true position but isn't exactly it (that gap is exactly the error being
   corrected).
2. **Continuous touch-assisted correction**, matching what was actually validated
   above. Simpler to reason about (matches the tested numbers directly), but is a real
   architectural departure — touch would, indirectly, be influencing control through a
   corrected vision signal, not staying a fully independent evaluation reference.

**This choice needs your explicit decision**, not an assumption — it's a
CLAUDE.md-relevant design question, not a pure engineering detail.

## What's needed before this touches hardware

1. **Re-validate variant 1** (fit-on-touch, apply-on-vision) specifically — the
   current numbers are oracle-only, per the section above.
2. **`TargetStateMachine` has no way to command an arbitrary `(x_mm, y_mm)` target
   yet** — it only supports `center`, `hold`, or a live marker-color average. Driving
   the ball through `generate_calibration_grid()`'s 9 points needs a new target-setting
   capability (or a manual/scripted physical placement procedure), not yet built.
3. **Not wired into `run_jetson_standalone.py`.** The insertion point is
   `JetsonExpertPolicy.act()`, right after `px_to_touch_mm()` produces `raw_x, raw_y`
   and before `PredictionGate.filter()` — see that file for the exact call site. Not
   done here on purpose: this is a new parametrized layer on top of CLAUDE.md's
   **LOCKED** "ArUco homography → warped → pixel × scale = mm (no MLP)" coordinate
   mapping decision (a simple affine fit, not an MLP, but still an addition to that
   pipeline) and needs your explicit sign-off first, not a unilateral change.
4. **No standing global-correction workflow defined.** `fit_global_correction()`
   computes a global fit from a set of past sessions' samples; nothing here decides
   when that gets re-run or how it's persisted across sessions.
5. **Blending's benefit needs re-confirming** (see "NOT robust" above) before its
   default alpha is trusted.

## Root-causing *why* the session-to-session geometry varies (separate from the fix above)

Not yet tested — needs hardware, in this priority order (fastest/most diagnostic
first):

1. **Same-process, repeated calibration, no restart.** Run the grid calibration three
   times within one continuous session (start/middle/end), nothing touched. Stable
   coefficients across all three would rule out the camera/process itself as the
   cause.
2. **Back-to-back restarts, nothing touched.** Stop/restart the process 2-3 times,
   hands off the camera and sheet, recalibrate fresh each time. If coefficients still
   shift, that implicates software/camera reinitialization (auto-exposure/focus
   reacquiring on `cv2.VideoCapture` open) — matching what the `ml-vision` subagent
   already guessed (not confirmed) for one session's outlier behavior.
3. **Deliberate, measured physical nudge** between two runs, then recalibrate — checks
   whether the shift direction/magnitude tracks a known physical perturbation.
4. Only after 1-3: **multi-day recording**, to isolate a slower environmental effect
   (lighting, thermal) layered on top of whatever the fast tests already find.

## Files

- `host_software/ml_jetson_vla/core/vision_calibration.py` — the correction math
  (fit/apply/blend/grid/collector). Not imported by any production entry point.
- `host_software/ml_jetson_vla/deployment/validate_vision_calibration.py` — reproduces
  the validated numbers from a pooled per-frame CSV (`session, touch_x, touch_y,
  vision_x, vision_y` columns). Run 2 in the results table above is this script's
  output.
