# Touch-staleness analysis (Track 4, 10 sessions) — 2026-10-07

Offline analysis on Windows. No Jetson access, no writes under `host_software/data/`, no commits.
Staleness flag reused verbatim from `compute_touch_stale` in
`host_software/ml_jetson_vla/data_processing/convert_to_lerobot.py` (extracted by AST and executed, not redefined).
Scripts live in the session scratchpad (`touch_stale.py`, `speed.py`, `longrun.py`).

## Data and column check

- Sessions: the 10 `session_jetson_track4_*` directories with a real `telemetry.csv` (the `.dvc` pointer files were excluded).
- Columns confirmed from the header: `frame_index, host_timestamp_ms, target_x, target_y, touch_x, touch_y, theta_a, theta_b, theta_c, audio_command`.
- Timestamp: `host_timestamp_ms` (integer ms). Frame interval median is 40-41 ms, so the unit is consistent with a ~24-25 fps loop.
- No NaN touch values in any Track 4 session.
- PID sessions: `01_bronze` contains six non-Track4 `session_*` directories (20260728, 20260730, four from 20260810), not five as stated in the brief. Skipped as requested.
- Units of `touch_x/y` and `theta_*` are not stated in the converter or telemetry header. Touch magnitudes (about ±90) suggest mm, but this is **not confirmed**. The spec's "touch position" claims below are in whatever unit the file uses.

## 1. Per-session results

Run length = consecutive flagged frames. Hold = time from the last changed value before the run to the last stale frame of the run (ms to s).

| Session | Frames | Stale | Stale frac | Runs | Run len median / p90 / max (frames) | Longest hold (s) |
|---|---|---|---|---|---|---|
| 151627 | 2596 | 782 | 0.301 | 500 | 1 / 3 / 21 | 0.86 |
| 153053 | 2294 | 740 | 0.323 | 447 | 1 / 3 / 31 | 1.27 |
| 153247 | 2370 | 728 | 0.307 | 452 | 1 / 3 / 23 | 0.93 |
| 160025 | 5057 | 1556 | 0.308 | 976 | 1 / 3 / 49 | 2.00 |
| 160509 | 4747 | 1471 | 0.310 | 909 | 1 / 3 / 22 | 0.89 |
| 161239 | 5193 | 1544 | 0.297 | 977 | 1 / 3 / 25 | 1.02 |
| 161712 | 4946 | 1375 | 0.278 | 873 | 1 / 3 / 35 | 1.41 |
| 163702 | 3535 | 1272 | 0.360 | 639 | 1 / 3 / 108 | **4.77** |
| 164115 | 4856 | 1435 | 0.296 | 880 | 1 / 3 / 35 | 1.43 |
| 164743 | 5119 | 1575 | 0.308 | 1003 | 1 / 3 / 30 | 1.22 |

Pooled: 12,478 stale of 40,713 frames (0.307), 7,656 runs in total. (The speed analysis in section 3 uses the 40,417 frames that have a speed estimate, stale fraction 0.302 there.) Hold median 44 ms (one frame), p90 124 ms, p99 208 ms, max 4.77 s.

Other per-session facts:
- Touch-value change interval (ms between successive distinct values): median 43 ms in every session, p90 85-122 ms.
- Frame interval median 40-41 ms, p90 45 ms (49 ms in 163702).

## 2. Joint repeat with theta (snapshot test)

- Share of stale frames where all of theta_a/b/c are also unchanged: **6.3%** pooled (per session 4.0-5.7%, except 163702 at 19.5%). Non-stale frames: 1.6%.
- theta_a alone unchanged on stale frames: 12% (29% in 163702).
- When touch changes, theta changes too in 98.5% of frames. theta changes on about 95% of all frames.
- By run length (pooled):

| Run length (frames) | Runs | theta unchanged, mean share of run |
|---|---|---|
| 1-2 | 4,868 | 0.076 |
| 3-4 | 2,645 | 0.034 |
| 5-11 | 130 | 0.054 |
| 12-23 | 4 | 0.166 |
| ≥24 | 9 | 0.279 |

The nine long runs (≥24 frames, about 1 s or more) are mostly in 163702 (108, 56, 50 frames), plus 160025 (49), 161712 (35), 164115 (35). In those, theta repeats along with touch. Short runs do not.

## 3. Clustering and position

- Stale fraction by session quarter is roughly flat (about 0.26-0.37 in each quarter). The exception is 163702, which rises to 0.42 in the last quarter.
- Autocorrelation: P(stale | previous frame stale) = 0.373 vs P(stale | previous fresh) = 0.271. Weak clustering.
- Touch position, stale vs non-stale frames (mean distance from origin, sessions in the same order): 28.2 vs 27.3; 32.4 vs 32.1; 34.5 vs 31.9; 35.8 vs 33.8; 33.3 vs 32.0; 18.5 vs 18.4; 20.8 vs 20.4; 46.7 vs 39.9; 28.1 vs 27.5; 33.4 vs 32.3. Stale frames sit slightly further from centre in all ten sessions, by about 1-3 units, with 163702 the largest gap.
- Stale fraction vs estimated ball speed (speed = displacement / time between the last two distinct touch values, applied to the following frames):

| Est. speed (units/s) | Frames | Stale frac |
|---|---|---|
| 0-5 | 1,369 | 0.239 |
| 5-10 | 3,853 | 0.255 |
| 10-20 | 9,288 | 0.283 |
| 20-40 | 14,477 | 0.298 |
| 40-80 | 8,682 | 0.321 |
| 80-160 | 1,964 | 0.389 |
| ≥160 | 784 | 0.499 |

Stale fraction **rises** with speed. The speed estimate is noisy and partly confounded by the stale runs themselves, but the direction is opposite to what a pure "ball is still" explanation predicts.

- Audio command: per-session stale fraction ranges 0.21 (stop, 161712) to 0.57 (left, 163702). I did not pool stale fraction by command across sessions, so no pooled per-command table is given. The high `left` value in 163702 is the long-run session.

## 4. Between-session variation

- Stale fraction: min 0.278 (161712), median about 0.307, max 0.360 (163702). Spread about 8 points.
- 151627 (the session the Arm 2 evaluation found hard) is at **0.301, essentially the median**. It is not a stale-fraction outlier. Its longest run is 21 frames (0.86 s), also typical.
- 163702 is the outlier: highest stale fraction, the 108-frame (4.77 s) run, and the only session with substantial joint theta/touch repeat (19.5%).

## 5. Quantisation

- Distinct touch_x values per session: 306-563. Minimum nonzero step between successive distinct values: 0.2 in every session. Mode step: 0.21, not exactly 0.2, so the grid is approximate. The sensor (or a float rounding stage) produces a quantum of about 0.2 units.
- A quantisation-only explanation predicts stale frames concentrated at low speed (displacement per frame below one quantum). Observed stale fraction at speed under 5 units/s is 0.24, lower than at high speed. This does not support quantisation as the dominant driver.

## 6. Verdict on cause

The data separate the three causes only partly.

- **Real ball stillness: not supported as the main driver.** Stale fraction is higher when the ball moves faster, and about 68% of stale frames are isolated single-frame repeats of a value that changes again next frame. A still ball would predict long runs.
- **Sensor quantisation: not supported as the main driver.** The grid is about 0.2 units, but stale repeats do not concentrate at low speed. Quantisation may contribute at the margin.
- **Async staleness:** this is the best fit for the **short runs** (about 90% of runs are 1-3 frames, hold median 44 ms, theta keeps moving, 94% of stale frames have theta changing). That pattern points to the touch channel updating on a cadence slightly slower than the loop (change interval median 43 ms against a 40-41 ms frame interval), independent of theta. It is **not** the shared-snapshot pattern the spec describes, because theta would repeat in lockstep and it almost never does on short runs.
- **The long tail (≥24 frames, rare, 9 runs, mostly 163702) does match a whole-snapshot stall:** theta repeats about 28% of the time within those runs. This is the documented async-staleness signature, but it is confined to a few episodes.

Net: the flag mixes a frequent, sub-frame touch-update lapse (async-like, theta unaffected) with rare whole-snapshot stalls. The data do not isolate the hardware or firmware mechanism behind the touch lapse. That would need the uplink/firmware timing, which is not in these logs.

## 7. Options for the user (no choice made here)

1. **Mask** `touch_stale == 1` frames from Stage 2/3 training. Removes about 31% of frames, including the rare long stalls. Biases the data toward moving-ball frames (stale rate rises with speed).
2. **Keep** all frames. Preserves the fast-motion samples but trains on repeated touch values as if current, which the spec flagged as risk #3.
3. **Keep but weight**: keep the frames, down-weight stale ones (or down-weight only runs of 24 or more frames, where theta is also stale). Retains coverage with a lower influence from the stalls.
4. A variant of option 3 is to mask only the long-run tail (≥24 frames, about 0.3% of frames), which is the only portion with a clear whole-snapshot signature.

## Limits

- Units of touch and theta not confirmed.
- Speed-binned result is confounded by the stale runs (speed is estimated from the last change interval) and is an estimate only.
- The six non-Track4 `session_*` directories were not inspected (PID sessions, skipped by request).
- The mechanism behind the touch update cadence is not confirmed from firmware or uplink logs.
