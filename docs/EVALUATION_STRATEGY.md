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

---

## Evaluation Pipeline

The evaluation is performed by `host_software/evaluations/evaluate_system_control.py`, which parses telemetry CSVs from any controller (PID baseline, the expert pipeline, or a VLA policy) containing `target_x/y`, `touch_x/y`, and `theta_a/b/c`, plus one of several recognized timestamp column names (`host_timestamp_ms`, `host_command_sent_ms`/`host_packet_received_ms`, or the legacy `host_time_ms` — see `TIMESTAMP_CANDIDATES` in the script; different eval scripts across the project's history have used different names for this column, so the loader normalizes across them rather than assuming one).

- **Single run:** `python evaluate_system_control.py --csv_path <telemetry.csv> --output_dir <dir>` — writes `control_metrics.json` + a trajectory plot.
- **Multi-run comparison (expert vs. baseline_vla vs. our_vla, etc.):** `python evaluate_system_control.py --runs expert=<csv> baseline_vla=<csv> our_vla=<csv> --report_dir <dir>` — writes one comparison JSON/CSV/bar-chart PNG across all labeled runs.

Task Success Rate currently measures "settled within tolerance before the trial ended" only — no telemetry schema in the project currently carries a ball-drop signal, so a dropped ball is not distinguished from a trial that simply never settled. Closing that gap requires an upstream drop-detection signal (e.g. from vision), not a change to this script.

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
