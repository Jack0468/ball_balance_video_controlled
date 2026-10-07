# Touch-sensor glitch-spike analysis — 2026-10-07

Windows-only, read-only investigation plus one converter addition. No Jetson access, no writes
under `host_software/data/` other than the required dry-run smoke-test dataset (created and then
deleted, per instructions). No commits/pushes. Pinned interpreter:
`C:/Users/Admin/.conda/envs/ball_balance_env/python.exe`.

Triggered by a real, already-confirmed finding: reviewing images in
`host_software/ml_jetson_vla/reports/ball_over_marker/`, 5 frames showed a rendered ball
position (from `touch_x`/`touch_y`) nowhere near the real ball. The raw telemetry rows around
each flagged frame show touch_x/touch_y jumping to an extreme value for 1-4 consecutive frames,
then snapping back to the prior trajectory — a transient sensor glitch, distinct from the
already-documented async staleness in `reports/touch_staleness_analysis_2026_10_07.md` (that
flag only ever catches the second-and-later frame of a glitch run, never the worst, first frame).

## Sessions covered

- **10 Track4 sessions**: the real `session_jetson_track4_*` directories under
  `host_software/data/01_bronze/` returned by `convert_to_lerobot.find_sessions()` (the `.dvc`
  pointer files that glob-match the same pattern were excluded — they are not directories).
  Total 40,713 frames (matches `session_manifest.json`'s `jetson_track4_total_frames`).
- **5 PID sessions**: exactly the sessions `convert_to_lerobot.find_pid_sessions()` returns by
  reading `session_manifest.json` (`regime == "pid_laptop_webcam"` AND `frame_synced_csv` is not
  null) — `session_20260728_102908`, `session_20260810_104132/110239/112047/114330`, each read
  via its `synced_telemetry.csv`. `session_20260730_174916` is excluded (never synced), matching
  the converter's own selection. Total 110,993 frames.

## Threshold and detector

A frame is a **glitch-spike candidate** if `touch_x` or `touch_y` changes by more than a
threshold from the frame before, **and** changes back by more than that threshold within the next
`K` frames. Jump magnitude = `max(|Δtouch_x|, |Δtouch_y|)` between consecutive rows.

**Real Track4 jump-magnitude distribution** (40,703 consecutive-row diffs, 10 sessions): median
0.66, p90 3.01, p95 4.18, p99 10.69, p99.5 27.98, p99.9 95.07, max 130.81. The 60 largest values
form an **unbroken tail from ~85 to 130.81 with no gap larger than ~4.4** between consecutive
sorted values — there is no magnitude-only elbow separating "fast legitimate ball motion" from
"glitch." The 5 manually-confirmed real examples all have an entry/return jump of 91.6-98.8, i.e.
they *are* the extreme tail, but magnitude alone can't be used to draw a clean line below them
without also risking a line that would include ordinary fast motion, because the distribution is
continuous there.

**Threshold chosen: `GLITCH_JUMP_THRESHOLD_MM = 30.0`** (≈ the 99.52nd percentile of the real
distribution). This is comfortably below the smallest entry/return jump in any of the 5 known
examples (91.55) — so none are missed on magnitude grounds — and far above the p99 (10.69) bulk of
real motion. Because the tail has no gap, **the real discrimination is done by the compound
condition** (an entry jump answered by a comparable return jump within `K` frames), not by the
threshold in isolation; a lone large jump that never reverts (e.g. real fast continuous motion) is
never flagged.

**K (look-ahead window) chosen: 5.** Swept at K=3/5/8:

| | K=3 | K=5 | K=8 |
|---|---|---|---|
| Track4 runs / frames | 83 / 118 | 85 / 128 | 85 / 128 |
| PID runs / frames | 137 / 161 | 136 / 174 | 136 / 174 |

K=5 and K=8 are **identical** on this corpus — no real run exceeds 5 frames beyond what K=5
already resolves, so widening further adds nothing. K=3 **misses one of the 5 known examples**
(see cross-check below) and undercounts by 10 Track4 frames and several PID frames, because some
real glitch runs last longer than 3 frames and the return jump falls outside a 3-frame look-ahead.

**Boundary handling (first/last K rows):** the first row of every session is always 0 (no prior
row to jump from, same convention as `touch_stale`). Near the end of a session, if an entry jump
occurs with fewer than `K` frames remaining, the look-ahead window is simply shorter and whatever
return would have been found with more data cannot be — treated conservatively as "not flagged"
(the same outcome as a jump that genuinely never returns), rather than guessing.

## Cross-check: the 5 known examples (K=5, threshold=30)

All 5 are caught:

| Session | frame_index | Run (0-based positions) | Entry jump | Return jump |
|---|---|---|---|---|
| …151627 | 1750 | 1750-1751 (2 frames) | 98.83 | 94.06 |
| …153247 | 958 | 958-958 (1 frame) | 98.42 | 98.21 |
| …161239 | 4342 | 4342-4343 (2 frames) | 98.21 | 94.06 |
| …164115 | 2326 | 2326-2330 (5 frames) | 97.23 | 96.39 |
| …164743 | 1990 | 1990-1992 (3 frames) | 93.22 | 91.55 |

**At K=3, the …164115/2326 example is MISSED.** Its run is 5 frames long (2326-2330); the return
jump lands at frame 2331, which is `entry_frame + 5`, outside K=3's look-ahead of `entry_frame +
1` through `entry_frame + 3`. K=5 (and K=8) catch it because the look-ahead reaches far enough.
This is why K=5, not K=3, is used as the converter's flag.

## Task 1 results: prevalence

**Track4 (10 sessions), K=5, threshold=30mm:**

| Session | Frames | Glitch runs | Glitch frames | Fraction | Caught by touch_stale |
|---|---|---|---|---|---|
| 151627 | 2596 | 7 | 8 | 0.308% | 1/7 |
| 153053 | 2294 | 3 | 8 | 0.349% | 2/3 |
| 153247 | 2370 | 9 | 12 | 0.506% | 2/9 |
| 160025 | 5057 | 8 | 10 | 0.198% | 2/8 |
| 160509 | 4747 | 10 | 13 | 0.274% | 3/10 |
| 161239 | 5193 | 10 | 15 | 0.289% | 4/10 |
| 161712 | 4946 | 13 | 18 | 0.364% | 3/13 |
| 163702 | 3535 | 4 | 5 | 0.141% | 1/4 |
| 164115 | 4856 | 11 | 22 | 0.453% | 4/11 |
| 164743 | 5119 | 10 | 17 | 0.332% | 5/10 |
| **Pooled** | **40,713** | **85** | **128** | **0.314%** | **27/85 (31.8%)** |

Run-length distribution (85 runs): **1 frame: 57, 2 frames: 17, 3 frames: 9, 5 frames: 2** — the
large majority (57/85, 67%) are single-frame, matching the "mostly 1-4 frames" expectation from
the 5 known examples; the two 5-frame runs are the longest observed.

**27 of 85 runs (31.8%) have at least one frame already caught by `touch_stale`** (because the
glitched value happens to repeat verbatim for ≥2 frames in those runs); **58 of 85 (68.2%) are
entirely missed by `touch_stale`** — confirming the motivating claim that `touch_stale` only ever
catches the second-and-later frame of a subset of glitch runs, never the worst/first frame, and
misses the majority of runs outright (every single-frame run, 57 of them, is invisible to
`touch_stale` by construction, since a 1-frame glitch never repeats).

**PID (5 sessions), K=5, threshold=30mm:**

| Session | Frames | Glitch runs | Glitch frames | Fraction | Caught by touch_stale |
|---|---|---|---|---|---|
| 20260728_102908 | 7,913 | 14 | 19 | 0.240% | 1/14 |
| 20260810_104132 | 28,794 | 30 | 39 | 0.135% | 8/30 |
| 20260810_110239 | 20,850 | 24 | 28 | 0.134% | 1/24 |
| 20260810_112047 | 26,677 | 34 | 40 | 0.150% | 2/34 |
| 20260810_114330 | 26,759 | 42 | 48 | 0.179% | 5/42 |
| **Pooled** | **110,993** | **136** | **174** | **0.157%** | **17/136 (12.5%)** |

Run-length distribution (136 runs): **1 frame: 115, 2 frames: 14, 3 frames: 2, 4 frames: 3, 6
frames: 1, 7 frames: 1.** PID glitch runs are slightly more frequent in count than Track4's but a
smaller fraction of frames overall (0.157% vs 0.314%), and `touch_stale` catches a smaller share
of them (12.5% vs 31.8%).

## Does the commanded motor angle (theta) also spike?

**Track4: no, theta stays isolated from the touch glitch.** Across all 10 sessions, `theta_a/b/c`
**never** exceeds a 20°/frame change anywhere in the corpus (0 of 40,703 frame-to-frame deltas),
glitch or not — the theta channel's dynamic range in Track4 is simply too small for a 20°+ jump to
occur under any circumstance. 0 of 85 glitch runs show a large theta delta on entry. This matches
qualitative inspection of all 5 known examples (e.g. session 151627, frame 1750: theta values stay
in the ±4° range through the entire glitch, changing no more than they do on an ordinary frame).
**Conclusion for Track4: the glitch is confined to `observation.state`/`target` (touch-derived);
`action` (theta) is not corrupted on these frames.**

**PID: yes, theta frequently co-spikes with the touch glitch.** Baseline rate: a >20°
frame-to-frame theta change happens on only 257 of 110,988 frames overall (0.23% — this is
approximately the 99.77th percentile of PID's own theta-delta distribution, so it is genuinely
rare in general). But **81 of 136 glitch runs (59.6%) show a >20° theta jump on the very frame the
touch glitch starts** — roughly 260x the baseline rate. Spot-checking the raw rows confirms this
is real, not a sampling artifact: e.g. `session_20260728_102908` row 856-857, `touch_x/y` jumps to
`104.34/81.53` for 2 frames while `theta_a/theta_b/theta_c` simultaneously jump to
`26.44/-21.15/17.44` from a baseline near `9.45/10.01/9.23`, then both snap back together.
**Conclusion for PID: the glitch is NOT isolated to touch there — `action` (theta) is jointly
corrupted on the same frames in the majority of PID glitch runs.** This is a materially different
failure mode from Track4's and is worth flagging separately to anyone using the PID regime's
`action` labels for training.

## Task 2: converter changes and test results

Added to `host_software/ml_jetson_vla/data_processing/convert_to_lerobot.py`, mirroring
`compute_touch_stale`'s existing pattern:

- `GLITCH_JUMP_THRESHOLD_MM = 30.0`, `GLITCH_LOOKAHEAD_K = 5` module-level constants (justified
  above).
- `compute_touch_glitch(df: pd.DataFrame) -> np.ndarray` — same NaN-never-sets-or-absorbs-the-flag
  rule as `compute_touch_stale`, first row always 0, end-of-session boundary treated
  conservatively (see above).
- New per-frame feature `"touch_glitch"` (`dtype: "int64"`, `shape: (1,)`) added to
  `_build_features()`, right after `"touch_stale"`.
- `glitch_flags = compute_touch_glitch(df)` computed alongside `stale_flags`, and
  `"touch_glitch": np.array([glitch_flags[pos]], dtype=np.int64)` added to the `add_frame()` dict,
  same indexing pattern (`pos`) as `touch_stale`.
- `touch_stale` itself is unchanged; no rows are filtered or dropped — only flagged, per
  instructions.

**Dry-run smoke test**: `python convert_to_lerobot.py --out-root
host_software/data/lerobot_dryrun_glitchcheck --max-frames-per-session 2500` (all 10 Track4 + 5
PID sessions, 2500-frame cap — large enough to include known glitch frames 958, 1750 and 1990).
This run is CPU-bound by SVT-AV1 video encoding (roughly 20-40 min/episode on this machine for a
2500-frame episode) — it was still mid-run after ~40 minutes (2 of 15 episodes' videos written)
when it was superseded by a faster, equally real check below and then stopped; the background
process did report a clean exit (code 0) around the time it was stopped, but the output directory
was already removed as part of cleanup before that could be separately re-inspected, so this
report does not claim to have manually verified that specific full-corpus run's on-disk output.

Instead, the required verification (feature exists, is 0/1, and the two named frames are flagged)
was obtained through **two independent, equally real checks against the identical, installed
`convert_to_lerobot.convert()` production code path**, both faster because they scope to only the
sessions actually needed:

1. Two small, targeted real end-to-end conversions, each using the real `convert()` function
   (same `LeRobotDataset.create`/`add_frame`/`save_episode`/`finalize` calls, same video encoder)
   restricted via `--session-pattern`/`--exclude-pid`/`--max-frames-per-session` to exactly the
   one session each named frame lives in:
   `--session-pattern session_jetson_track4_20260915_151627 --exclude-pid
   --max-frames-per-session 1760` and `--session-pattern session_jetson_track4_20260915_153247
   --exclude-pid --max-frames-per-session 970`. Reading the resulting parquet back directly:

   ```
   151627 rows=1760 has touch_glitch col: True dtype=int64
     frame_index 1750 -> touch_glitch=1 touch_stale=0 value set=[0, 1]
   153247 rows=970  has touch_glitch col: True dtype=int64
     frame_index 958  -> touch_glitch=1 touch_stale=0 value set=[0, 1]
   ```

   Confirms exactly what Task 2 asked to verify: the feature exists, is strictly 0/1 `int64`, and
   both named frames are flagged `1` — while `touch_stale` is `0` at both of those exact frames,
   independently confirming on-disk (not just in the Task 1 analysis script) that `touch_stale`
   misses what `touch_glitch` catches.
2. `test_t4_end_to_end_synthetic_session` (below) runs the same real `convert()` function on a
   small synthetic session and asserts the on-disk `touch_glitch` column matches expected flags
   exactly, plus that `touch_stale` is unaffected.

Both checks exercise the literal, installed code added in this task, not a reimplementation or
a mock. The large 15-session/2500-frame run was a slower superset of the same code path and
was not needed once these confirmed the result; `host_software/data/lerobot_dryrun_glitchcheck`
was deleted, leaving no residue under `host_software/data/`.

**Unit tests** — added to `host_software/ml_jetson_vla/tests/test_lerobot_frame_roundtrip.py`,
mirroring the existing `touch_stale` (T3) tests exactly in style:

- `test_t4_compute_touch_glitch_pure` — realistic drift-spike-return shape (matches the 5 known
  examples' shape), a single-frame spike, and a large jump that never returns (not flagged).
- `test_t4_compute_touch_glitch_nan_and_boundary` — NaN rows never set/absorb the flag; an entry
  jump too close to the end of the session (truncated look-ahead) is not flagged.
- `test_t4_end_to_end_synthetic_session` — runs the real `convert()` on a synthetic session,
  reads the parquet back, and asserts `touch_glitch` matches expectations exactly, that the
  feature is declared in `_build_features()`, and that `touch_stale` is unaffected.

**Full test file run** (pinned interpreter, `python
ml_jetson_vla/tests/test_lerobot_frame_roundtrip.py`):

```
PASS  test_a_converter_state_is_raw_touch
PASS  test_b_scorer_roundtrip_manifest_points
PASS  test_b2_scorer_roundtrip_converter_rows
PASS  test_c_state_frame_matches_scorer_convention
PASS  test_t3_compute_touch_stale_pure
PASS  test_t3_end_to_end_synthetic_session
PASS  test_t4_compute_touch_glitch_pure
PASS  test_t4_compute_touch_glitch_nan_and_boundary
PASS  test_t4_end_to_end_synthetic_session

9/9 passed
```

## Limits

- Threshold/K were chosen from this corpus; they are not guaranteed to generalize to future
  sessions with different motion dynamics without re-checking the distribution.
- The compound entry+return detector can, by construction, treat the *return* frame of one run as
  the *entry* of a search for a further return (the algorithm resumes scanning from the return
  frame rather than skipping past it). This never produced a false additional run anywhere in the
  real 15-session corpus (verified by direct inspection) but is a theoretical edge case worth
  knowing about if the data distribution changes.
- PID's `theta` co-spike finding is based on a fixed 20°/frame threshold chosen against PID's own
  baseline distribution; it is not necessarily the right threshold for a different regime.
- The large (2500-frames-per-session) end-to-end dry-run through the real `LeRobotDataset`/AV1
  video pipeline is slow on this machine (SVT-AV1 encoding is CPU-bound); the correctness evidence
  in this report relies primarily on the direct-function-call and small-synthetic-session checks,
  which exercise the identical, real, installed code path.
