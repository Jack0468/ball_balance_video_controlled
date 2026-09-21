# System-Level Evaluation Strategy

This document formalizes the evaluation criteria for the entire VRI 2026 Ball Balancing Robot system. To compare classical control (PID), ML-augmented control (Expert Vision + MLP/RL), and end-to-end approaches (VLA), we analyze hardware-level telemetry logs instead of isolated component metrics.

## 1. Steady-State Error (Euclidean mm)
**Definition:** The average Euclidean distance between the ball and the target coordinate after the system has settled.
**Formula:** `sqrt((touch_x - target_x)^2 + (touch_y - target_y)^2)`
**Significance:** Measures absolute balancing accuracy. Since the vision system has inherent noise (RMSE of ~4-5mm), achieving a Steady-State Error below 10mm implies near-perfect real-world control.

## 2. Settling Time (ms)
**Definition:** The time elapsed from when a new target command is issued until the ball enters and remains within a stable 20mm radius of the target for at least 500ms.
**Significance:** Measures system responsiveness and overshoot damping. A fast settling time indicates an aggressive but well-tuned controller.

## 3. Control Effort (Theta Jerk/Variance)
**Definition:** The absolute sum of differences in motor servo angles (`theta`) between consecutive frames over the evaluation period.
**Formula:** `Σ |theta(t) - theta(t-1)|` for servos A, B, and C.
**Significance:** Measures control efficiency. A lower effort indicates smooth motor actuation, which reduces power consumption, mechanical wear, and heat. High effort (jitter) implies a noisy controller or over-reaction to sensor noise.

## 4. Task Success Rate (%)
**Definition:** The percentage of discrete trials (target shifts) where the ball successfully reaches the target region without being dropped off the platform within the allocated timeout period.
**Significance:** The ultimate binary indicator of system reliability.

## Extended Metrics (added 2026-09-18, not part of the original four)

Standard time-domain and integral-error metrics from classical control theory, computed by the same `compute_metrics()` in `evaluate_system_control.py` as the four metrics above, from the same telemetry — no new recording needed. Added after a report audit found these are commonly tracked for control-system/ball-balancing evaluation but weren't yet part of this pipeline.

- **Rise Time (ms):** time for the scalar Euclidean error to drop from its trial-start value to 10% of that value. **Not** the classical 10%-90%-of-final-value definition — that assumes a *rising* step response, while "distance to target" here only ever decreases, so the definition is adapted accordingly. Trials starting within 1mm of target (nothing to rise from) or that never reach the 10% threshold are excluded from the average, not counted as 0.
- **Overshoot (mm and %):** per-axis (signed X/Y error, not the scalar distance used elsewhere — overshoot needs a direction to overshoot past), the peak excursion to the *opposite* side of the target after the signed error first crosses zero. Reported both as an absolute mm value and as a percentage of the trial's own initial error. **The percent version is unreliable for trials with a small initial error** — confirmed on real data, a trial starting ~1mm from target produced a 462% "overshoot" from perfectly ordinary millimeter-scale oscillation. Prefer the mm value; the percent version is kept for comparability with classical control literature, not as the primary number.
- **IAE / ISE / ITAE (Integral Absolute/Squared/Time-weighted-Absolute Error):** standard integral performance indices, trapezoidal-integrated over each trial's actual (jittery) sample timestamps. Computed over the trial's **full duration**, not just until settle — unlike Steady-State Error and Settling Time, these don't require the trial to have succeeded, so every trial with ≥2 samples contributes a value. ITAE weights late-persisting error more heavily than early (unavoidable) error, which the plain IAE/ISE indices don't distinguish.

Not yet added, flagged for later: ball-trajectory jerk/smoothness (distinct from Control Effort above, which is motor-angle-based, not ball-position-based) and disturbance-rejection/robustness metrics under injected perturbation — both came up in the same audit as standard for RL-controlled systems but need either new instrumentation or a dedicated test protocol, not just a new formula over existing telemetry.

---

## Evaluation Pipeline

The evaluation is performed by `host_software/evaluations/evaluate_system_control.py`, which parses telemetry CSVs from any controller (PID baseline, the expert pipeline, or a VLA policy) containing `target_x/y`, `touch_x/y`, and `theta_a/b/c`, plus one of several recognized timestamp column names (`host_timestamp_ms`, `host_command_sent_ms`/`host_packet_received_ms`, or the legacy `host_time_ms` — see `TIMESTAMP_CANDIDATES` in the script; different eval scripts across the project's history have used different names for this column, so the loader normalizes across them rather than assuming one).

- **Single run:** `python evaluate_system_control.py --csv_path <telemetry.csv> --output_dir <dir>` — writes `control_metrics.json` + a trajectory plot.
- **Multi-run comparison (expert vs. baseline_vla vs. our_vla, etc.):** `python evaluate_system_control.py --runs expert=<csv> baseline_vla=<csv> our_vla=<csv> --report_dir <dir>` — writes one comparison JSON/CSV/bar-chart PNG across all labeled runs.

Task Success Rate currently measures "settled within tolerance before the trial ended" only — no telemetry schema in the project currently carries a ball-drop signal, so a dropped ball is not distinguished from a trial that simply never settled. Closing that gap requires an upstream drop-detection signal (e.g. from vision), not a change to this script.

### Inference Latency (`vision_inference_ms`)

Added 2026-09-18 to both live-evaluation telemetry (`src/touch_logger.py`'s `CSV_FIELDS`) and Track 4 session telemetry (`ml_jetson_vla/runtime/session_recorder.py`'s `TELEMETRY_CSV_FIELDS`). Measured in `run_jetson_standalone.py`'s main loop as the isolated wall-clock time of the policy's `act()` call (ArUco homography + warp + CNN forward pass + marker classification + PredictionGate/Kalman + state-machine update) — deliberately **not** the same thing as `rtt_ms` (serial round-trip to the STM32) or the loop's `total_ms` (which also includes camera-frame wait and serial I/O). **Consumed by `evaluate_system_control.py`** (updated 2026-09-22; an earlier version of this paragraph said it wasn't consumed by anything yet): `compute_metrics()` adds `Mean_/Median_/P95_/Max_Inference_Time_ms` to a run's metrics whenever its CSV has the column, so they appear in the single-run `control_metrics.json` and the `--runs` comparison JSON/CSV, and `plot_comparison()` draws an extra inference-time panel (bar = median, whisker = P95) when at least one compared run has the data. Its purpose is a like-for-like comparison of the small-class expert pipeline's inference latency against a future large-model arm on the same Jetson hardware. A run without the column is omitted from these statistics — never averaged in as 0ms — and shows as "no data" in the chart. Optional/blank on any telemetry recorded before this date, and on any caller that doesn't pass `inference_ms`/`vision_inference_ms` — existing consumers reading required columns by name are unaffected.

**What the number does and doesn't cover — read before comparing against another arm** (added 2026-09-22, from reading the runtime rather than from measurements):
- It covers only the small pipeline's vision policy call, on the Jetson CPU (`onnxruntime` `CPUExecutionProvider`). It excludes: audio-model inference (its own background thread on a ~0.2s window cadence — `latest_inference_time_ms` exists on the receiver but is not logged), the RL control net (runs on the STM32 in Phase A, so it is not measured from the Jetson at all), camera capture (only printed as `[camera]` console lines), and the serial round-trip (`rtt_ms`, logged separately). Don't sum these into a single "system latency" figure without stating which parts are included.
- Cold start is generally **not** in the CSV. `act()` returns early on `no_aruco`/`no_ball` frames, and those frames are never logged, and the ball must be seen for `--seed-window` consecutive consistent frames before logging starts — so ONNX session load and the first several `act()` calls fall outside the recorded data. If cold-start latency matters for a comparison (it will for a large model), it has to be measured separately.
- `--eval-sequence` and `--dummy-audio` runs construct no audio model, so no audio thread competes for CPU during them; their vision latency may be optimistic relative to a live-audio deployment. This effect has not been measured.
- Rows are telemetry-echo rows (~25 Hz), not strictly one per vision frame, so the statistics are echo-weighted.
- `Max_Inference_Time_ms` includes any scheduling or warm-up transient in the logged frames; prefer median and P95 for comparisons. No warm-up trimming is applied — whether one is needed should be decided from a real run's first logged samples, not assumed.

### VLA Goal Alignment (Reinforcement Learning)
To ensure the end-to-end VLA model respects these criteria, we utilize a two-stage training process:
1. **Behavioral Cloning (BC):** Pre-trains the model to mimic the expert PID/RL outputs to achieve baseline balancing.
2. **Reinforcement Learning (PPO):** Fine-tunes the VLA policy using a reward function that explicitly optimizes for the evaluation triad:
   - **Reward:** Minimizing Euclidean Distance (Steady-State Error)
   - **Penalty:** High action variance (Control Effort / Jerk)
   - **Terminal Penalty:** Dropping the ball (Task Success Rate)

## Standardized Evaluation Sequence
To rigorously benchmark all systems (PID, Expert Vision+RL, VLA) under perfectly identical conditions, evaluations must be run against a pre-recorded audio sequence.

**Updated 2026-09-15: only `green`/`red`/`yellow`/`black` are used as target colors** (`grey`/`blue`
dropped) -- matches the physical marker sheet actually in use; `grey` and `blue` aren't reliable
targets on this sheet (see `ml-vision`'s marker-classification findings, session 2026-09-15).

**Updated 2026-09-15: `HOLD` placement redesigned.** Color-marker positions are physically movable
(placed on the board by hand, not fixed in `ground_truth_manifest.json` -- see
`PROJECT_LOGBOOK.md`'s "Movable Target Detection" entry), so a "hold between two far-apart
targets" design would depend on the current physical layout and wouldn't stay reproducible across
setups. Instead, `HOLD` now fires **mid-transit**: `FORWARD`+`LEFT` drive the ball toward the
top-left of the platform, a color command is then issued (driving it back across the board), and
`HOLD` interrupts that transit before it arrives -- tests whether the controller correctly holds
position mid-trajectory rather than only at a settled target. `go_black` is deliberately not the
first command (an untested cold-start target).

The evaluation script injects the following 10 commands with background continuously being fed inbetween commands (each spaced by exactly 10 seconds of evaluation time):
1. `go_green` (0s - 10s)
2. `go_yellow` (10s - 20s)
3. `FORWARD` (20s - 30s) -- drive toward top of platform
4. `LEFT` (30s - 40s) -- drive toward top-left
5. `go_red` (40s - 50s) -- command toward red, ball is mid-transit from top-left
6. `HOLD` (50s - 60s) -- interrupt the transit, hold here
7. `go_black` (60s - 70s)
8. `RIGHT` (70s - 80s)
9. `BACKWARD` (80s - 90s)
10. `STOP` (90s - 100s)

The host script is responsible for aligning these logical commands to the equivalent 2D physical target coordinates depending on the robot's current configuration.
