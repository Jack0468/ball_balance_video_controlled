"""Ball-drop detection -- distinguishing "the ball fell off the platform" from "the
trial simply never settled" within Task Success Rate (`docs/TODO.md`'s "Task Success
Rate can't distinguish a dropped ball from a trial that never settled", flagged
2026-09-18, owner: ml-multimodal/orchestration; `docs/EVALUATION_STRATEGY.md`'s
"Evaluation Pipeline" section; `host_software/evaluations/evaluate_system_control.py`'s
own `Task_Success_Rate_Caveat` string).

**Chosen signal, and why the other two candidates in the task don't actually work**:

1. Vision-side ball position, tracked independently of markers (candidate 2 in the
   task): this DOES exist (`CLAUDE.md`'s "Ball head: heatmap -> spatial softmax ->
   soft-argmax (X, Y)", and `touch_logger.py`'s own `vision_x_mm`/`raw_vision_x_mm`
   columns are exactly this). But it is useless for drop detection by construction:
   `evaluate_system_control.py`'s own "Inference Latency" doc section confirms `act()`
   "returns early on no_aruco/no_ball frames, and those frames are never logged" --
   i.e. the one moment vision would need to say "I lost the ball" is exactly the
   moment nothing gets written to telemetry at all. Silence, not a signal.
2. A bounds/sentinel check on `touch_x`/`touch_y` (candidate 1's second half): checked
   directly against `TouchProbe.cpp` (`firmware/stm32_ml_control_and_vision/
   BallBalancingBot/TouchProbe.cpp`, `emit_record()`) -- when contact is lost it does
   NOT emit an out-of-bounds value or a sentinel. It explicitly "Hold[s] the last
   position, flag[s] it invalid" (`last_touch_valid = false`, `last_touch_x/y` left
   untouched). So a dropped ball's logged `touch_x`/`touch_y` is whatever the ball's
   last real position was -- typically still well inside the 187.5x142.0mm platform
   bounds (confirmed on all 3 real drop candidates found below: none of their held
   positions were anywhere near the physical edge). A bounds check on position alone
   would miss every real case.
3. **`touch_valid` (candidate 1's first half) is the real, working signal.**
   `TouchProbe.cpp` sets it from a 2-of-3-raw-samples pressure-vote, and -- critically
   -- `TOUCH_EMIT_WHEN_LOST=1` means an invalid reading is deliberately still logged
   ("Emit a record even when no ball is present. You want the dropouts in the log").
   This is the only "ball physically off the plate" signal this project's telemetry
   carries anywhere, and it already exists in `touch_logger.py`'s `CSV_FIELDS`
   (consumed today by `evaluate_system_control.py`'s own `--filter-touch-outliers`
   path, so the column is already plumbed as far as that script).

**Real-data audit (2026-10-14), all 15 sessions in
`host_software/data/01_bronze/evaluation/ground_truth_jetson_*.csv`** (the only
sessions in this repo whose schema carries `touch_valid` at all -- see
`Known gaps` below for the two schemas that don't): loaded through
`evaluate_system_control.py`'s own `load_telemetry()`/`segment_by_target()`
(imported read-only) and scanned for every trial segment ending in a trailing,
non-recovered `touch_valid==0` run. Two confounds had to be ruled out before trusting
the result:

  - **Leading-stale exclusion.** `touch_ground_truth_filter.py` already documents that
    the first ~0.4-2s of some recordings holds a STALE position from before the
    recording started, with `touch_valid` never having gone 1 yet. The very first
    segment of `ground_truth_jetson_20260922_103659.csv` (rows 0:252) is entirely
    `touch_valid==0` (10039ms, `had_prior_valid=False`) -- this is that same artifact,
    NOT a drop (nothing was ever confirmed on the plate to drop). Requiring at least
    one earlier `touch_valid==1` sample within the segment excludes this case
    correctly.
  - **Ordinary pressure-vote noise at a segment boundary.** Across all 15 sessions,
    every segment that ends while genuinely still tracked shows at most a 0-120ms
    trailing invalid run (coincidental single-frame echoes right at the boundary; 44
    segments at 0ms, 12 at ~40ms, 4 at ~80ms, 3 at ~120ms). There is a clean, large gap
    between this noise floor and the next data point.

  After excluding the leading-stale case, exactly 3 segments (out of ~210 total across
  all 15 files) show a trailing, non-recovered `touch_valid==0` run that both (a)
  followed real earlier contact and (b) is far above that noise floor:
  `ground_truth_jetson_20260915_163702.csv` segment 11 (4760ms),
  `ground_truth_jetson_20260922_103659.csv` segment 6 (2760ms), and
  `ground_truth_jetson_20260922_104113.csv` segment 4 (10239ms, the trial's AND the
  whole session's last row -- the recording stops while still invalid). All 3 come
  from sessions that are also anomalously short relative to the standard ~100s
  evaluation sequence (1613/2994/1301 and 3754 rows respectively, vs. 2400-5400 for
  the other 11 sessions) -- consistent with, though not proof of, an operator stopping
  an evaluation run early because the ball had actually left the platform.
  `MIN_DROP_DURATION_MS=1000.0` sits in the clean gap between the 120ms noise ceiling
  and the 2760ms lowest real candidate.

  **Honesty check, per this task's own instruction**: no session in this dataset
  carries an authoritative, human-confirmed "the ball was dropped at this timestamp"
  label. Nothing in this repo logs that event directly. The 3 flagged segments are the
  strongest available offline proxy -- a real, physically-motivated signal
  (sustained, non-recovered loss of touch contact after confirmed prior contact) that
  cleanly separates from every other segment in real data by over an order of
  magnitude -- but this is circumstantial corroboration, not a confirmed count of real
  drops. Final confirmation against an observed/logged real drop event is exactly the
  JETSON-HW-gated step `docs/TODO.md`'s own item defers to hardware; this module's
  offline validation goes as far as existing logged data honestly allows and no
  further.

**A second, independent finding surfaced by this same audit, broader than the
original TODO wording**: `find_settling_time_ms()` computes `error_mm` from
`touch_x`/`touch_y` without ever consulting `touch_valid`. Checked directly against
the 3 real drop candidates above: 2 of them (`163702` segment 11, `104113` segment 4)
are CURRENTLY COUNTED AS SUCCESSES by `evaluate_system_control.py` today (settle times
of 200ms and 601ms respectively), because the error briefly dipped under the 20mm
tolerance for 500ms earlier in the segment, before the ball was later dropped -- the
settle check has no way to know contact was later and permanently lost. So the real
defect is not only "a failure bucket that conflates two causes" -- a dropped ball that
happened to be near target just before being dropped can currently be miscounted as a
SUCCESS outright. See `Integration notes` below for why this module's recommended
wiring overrides the settle result, not just annotates it.

**Known gaps -- `touch_valid` is not available on every schema this project logs**:
  - `host_software/ml_jetson_vla/runtime/session_recorder.py`'s Track 4
    `TELEMETRY_CSV_FIELDS` does NOT include `touch_valid`, even though
    `SessionTouchTap._handle_line()` already parses it into its in-memory
    `get_latest_touch()` dict (`"valid": bool(valid)`) -- it is parsed and then
    silently dropped before ever reaching `telemetry.csv`. Wiring it through
    `SessionRecorder.log()`/`_run()` (one more field, mirroring how `touch_x`/`touch_y`
    already flow from `get_latest_touch()` into `log()`) is a real, concrete,
    low-effort follow-up this audit surfaced -- not done here (outside this task's
    scope, and `session_recorder.py` is this agent's own file, so it CAN be touched,
    but doing so was not asked for by this task and would need its own
    verification pass).
  - `data/04_evaluation_expert/expert_evaluation_run*.csv` (the expert-pipeline PID
    telemetry schema) has no `touch_valid`-equivalent column at all in its header
    (`host_timestamp_ms,mcu_micros,target_x,target_y,touch_x,touch_y,error_x,error_y,
    pitch,roll,theta_a,theta_b,theta_c,integral_x,integral_y,deriv_x,deriv_y`) --
    `firmware/ExpertEvaluationFirmware/TouchProbe.cpp` streams touch position through
    a different telemetry path than the `T,...` protocol and never exposes a validity
    flag to it. This function degrades gracefully (returns `None`/an explicit
    "unavailable" caveat) rather than guessing when `touch_valid` isn't present, same
    convention `_inference_latency_stats()` already uses in
    `evaluate_system_control.py` for `vision_inference_ms`.

**Why this lives here, not in `evaluate_system_control.py` itself**: same reason as
`trajectory_smoothness.py` (see its module docstring) -- this agent's write boundary is
`host_software/ml_multimodal/` and `host_software/ml_jetson_vla/` only;
`host_software/evaluations/` is read-only for this agent. Unlike jerk/smoothness,
though, wiring this in is NOT a pure trailing `**merge` -- it needs to override the
existing settle/success determination for a segment where a drop is detected (see the
masking finding above), so it needs a few lines inside `compute_metrics()`'s existing
per-segment loop, not just at the loop's end. See `Integration notes` at the bottom of
this file for the exact, minimal patch.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

# TouchProbe.cpp's own 2-of-3-raw-samples pressure-vote majority filter already
# absorbs single-frame contact noise. Checked against all 15 real
# ground_truth_jetson_*.csv evaluation sessions (2026-10-14): every segment that
# ends while the ball is genuinely still tracked shows at most a 0-120ms trailing
# touch_valid==0 run (a coincidental single-frame echo right at the segment
# boundary). The lowest real drop candidate found was 2760ms. 1000ms sits in the
# large, clean gap between those two numbers -- see module docstring's audit.
MIN_DROP_DURATION_MS: float = 1000.0


def segment_drop_duration_ms(
    seg_times_ms: np.ndarray,
    seg_touch_valid: Optional[np.ndarray],
    min_drop_duration_ms: float = MIN_DROP_DURATION_MS,
) -> Optional[float]:
    """Duration (ms) of a trailing, NEVER-RECOVERED loss-of-contact run at the end of
    one trial segment, or `None` if this segment does not look like a drop.

    Returns `None` (not a drop) when any of:
      - `seg_touch_valid` is `None` (this telemetry schema doesn't carry the column --
        see module docstring's "Known gaps") or empty.
      - The segment did not end on an invalid (`touch_valid==0`) sample -- contact was
        still held (or regained) right up to the segment's own end.
      - The segment was invalid for its ENTIRE duration, i.e. `touch_valid` was never
        1 anywhere earlier in the segment -- there is no confirmed prior contact to
        have been dropped FROM (the "leading-stale-value" artifact documented in
        `touch_ground_truth_filter.py`, not a drop; see module docstring's audit).
      - The trailing invalid run is shorter than `min_drop_duration_ms` -- ordinary
        pressure-vote noise at the segment boundary, not a sustained loss.

    Deliberately conservative: a drop that happens mid-segment and recovers before the
    segment ends is NOT flagged by this function (it only looks at the trailing run) --
    this undercounts rather than overcounts drops, which is the safer direction for a
    metric that currently undercounts them completely (see `Task_Success_Rate_Caveat`
    in `evaluate_system_control.py` today)."""
    if seg_touch_valid is None:
        return None
    tv = np.asarray(seg_touch_valid)
    n = len(tv)
    if n == 0 or tv[-1] != 0:
        return None

    # Walk back from the end to find where the trailing invalid run starts.
    k = n - 1
    while k >= 0 and tv[k] == 0:
        k -= 1
    # k is now the index of the last VALID sample before the trailing run (or -1 if
    # the whole segment is invalid).
    if k < 0 or not np.any(tv[: k + 1] == 1):
        return None  # no confirmed contact earlier in this segment -- not a drop

    times = np.asarray(seg_times_ms, dtype=np.float64)
    duration_ms = float(times[-1] - times[k + 1])
    if duration_ms < min_drop_duration_ms:
        return None
    return duration_ms


def is_likely_ball_drop(
    seg_times_ms: np.ndarray,
    seg_touch_valid: Optional[np.ndarray],
    min_drop_duration_ms: float = MIN_DROP_DURATION_MS,
) -> bool:
    """Boolean convenience wrapper around `segment_drop_duration_ms()`."""
    return segment_drop_duration_ms(seg_times_ms, seg_touch_valid, min_drop_duration_ms) is not None


def ball_drop_metrics_for_segments(
    times_ms: np.ndarray,
    touch_valid: Optional[np.ndarray],
    segments: List[Tuple[int, int]],
    settled_mask: List[Optional[bool]],
    min_drop_duration_ms: float = MIN_DROP_DURATION_MS,
) -> Dict[str, object]:
    """Resolves each trial segment into exactly one of three buckets -- Settled,
    Dropped, or Never-Settled-(and-not-dropped) -- and returns a dict in the same
    `Average_*`/`*_Caveat` style as `evaluate_system_control.py`'s other Extended
    Metrics, ready to merge into `compute_metrics()`'s return dict with `**`.

    `settled_mask` must be aligned 1:1 with `segments`: for each `(start, end)` pair,
    `True` if the caller's own `find_settling_time_ms()` returned a value for it,
    `False` if it returned `None` (ran the check, didn't settle), or `None` if the
    segment was too short to evaluate at all (`len(seg_times) < 2` in
    `evaluate_system_control.py`'s loop -- excluded from this breakdown the same way
    it's already excluded from `Steady_State_Error_Trials_Excluded`/`Rise_Time_Trials_
    Excluded` elsewhere in that file, not folded into either bucket here).

    A segment flagged as a drop by `segment_drop_duration_ms()` is moved into
    `Dropped_Ball_Trials` EVEN IF `settled_mask` says it settled -- seet the module
    docstring's "second, independent finding": the existing settle check has no way to
    know a trial it called successful was later, permanently dropped. This is a
    correction to `Task_Success_Rate_Percent`, not just an annotation alongside it."""
    if touch_valid is None:
        return {
            "Dropped_Ball_Trials": None,
            "Never_Settled_Trials": None,
            "Settled_Trials_Overridden_By_Drop": None,
            "Task_Success_Rate_Caveat": (
                "Measures 'settled in time' only -- this run's telemetry schema does "
                "not carry touch_valid (see ball_drop_detection.py's 'Known gaps'), so "
                "a dropped ball is still NOT distinguished from a trial that simply "
                "never settled for THIS run."
            ),
        }

    tv = np.asarray(touch_valid)
    times = np.asarray(times_ms, dtype=np.float64)

    dropped = 0
    never_settled = 0
    overridden = 0
    corrected_successes = 0
    for (start, end), ok in zip(segments, settled_mask):
        if ok is None:
            continue  # too short to evaluate -- excluded, not counted either way
        seg_drop_ms = segment_drop_duration_ms(times[start:end], tv[start:end], min_drop_duration_ms)
        if seg_drop_ms is not None:
            dropped += 1
            if ok:
                overridden += 1
            continue
        if ok:
            corrected_successes += 1
        else:
            never_settled += 1

    total_resolved = dropped + never_settled + corrected_successes
    return {
        "Dropped_Ball_Trials": dropped,
        "Never_Settled_Trials": never_settled,
        "Settled_Trials_Overridden_By_Drop": overridden,
        "Task_Success_Rate_Percent_Drop_Corrected": (
            100.0 * corrected_successes / total_resolved if total_resolved else None
        ),
        "Task_Success_Rate_Caveat": (
            "'Settled in time', drop-corrected using touch_valid (see "
            "ball_drop_detection.py): a trial whose touch contact was lost and never "
            "recovered is counted as Dropped_Ball_Trials, not as a success or a plain "
            "never-settled failure, even if the error happened to be within tolerance "
            "just before the drop. Task_Success_Rate_Percent above does NOT reflect "
            "this correction (unchanged for comparability with historical runs) -- "
            "use Task_Success_Rate_Percent_Drop_Corrected for the corrected figure. "
            f"min_drop_duration_ms={min_drop_duration_ms}."
        ),
    }


# ---------------------------------------------------------------------------------
# Integration notes for whoever wires this into evaluate_system_control.py
# (host_software/evaluations/ -- outside this agent's write boundary; this is a
# drop-in but NOT trailing-only patch, unlike trajectory_smoothness.py's jerk metric):
#
# 1. Before the per-segment loop in compute_metrics():
#        touch_valid_col = df["touch_valid"].to_numpy() if "touch_valid" in df.columns else None
#
# 2. Inside the loop, replace the segment's settled_mask bookkeeping: track one
#    Optional[bool] per segment (None for the existing `if len(seg_times) < 2: continue`
#    branch, True/False for the existing settle_ms is None check) into a list, e.g.
#    `settled_mask: List[Optional[bool]] = []`, appending as the loop already iterates.
#    No change to the existing `successes`/`settling_times`/`settled_state_errors`
#    accumulation is required by this step alone.
#
# 3. After the loop, alongside the existing
#    `**jerk_metrics_for_segments(times, touch_x, touch_y, segments)` line:
#        **ball_drop_metrics_for_segments(times, touch_valid_col, segments, settled_mask)
#
# 4. Per the module docstring's masking finding, consider also subtracting
#    `Settled_Trials_Overridden_By_Drop` from `successes` before computing
#    `Task_Success_Rate_Percent` itself (not just reporting the corrected figure
#    alongside it) -- left to that file's owner to decide given the "unchanged for
#    historical comparability" tradeoff noted above.
# ---------------------------------------------------------------------------------
