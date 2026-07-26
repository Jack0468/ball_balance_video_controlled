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

The evaluation is performed by `host_software/evaluations/evaluate_system_control.py`, which parses the standardized telemetry CSV (e.g. `labels_sequential.csv` or VLA outputs) containing `host_timestamp_ms`, `target_x/y`, `touch_x/y`, and `theta_a/b/c`.

### VLA Goal Alignment (Reinforcement Learning)
To ensure the end-to-end VLA model respects these criteria, we utilize a two-stage training process:
1. **Behavioral Cloning (BC):** Pre-trains the model to mimic the expert PID/RL outputs to achieve baseline balancing.
2. **Reinforcement Learning (PPO):** Fine-tunes the VLA policy using a reward function that explicitly optimizes for the evaluation triad:
   - **Reward:** Minimizing Euclidean Distance (Steady-State Error)
   - **Penalty:** High action variance (Control Effort / Jerk)
   - **Terminal Penalty:** Dropping the ball (Task Success Rate)
