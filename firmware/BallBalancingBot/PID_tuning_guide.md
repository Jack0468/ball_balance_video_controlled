### Standard Operating Procedure: Dual-Region PID Controller Tuning

**Objective:** To systematically tune a dual-zone Proportional-Integral-Derivative (PID) controller to achieve rapid slew rates (large signal) and precise steady-state settling (small signal) while preventing integral windup and mechanical instability.

**Scope:** This procedure applies to embedded robotic actuators utilizing two distinct PID parameter sets divided by an error threshold ($e_{th}$).

**Prerequisites:**

* The control loop must operate at a consistent, deterministic time step ($\Delta t$).
* Sensor data must be appropriately filtered to prevent derivative kick.
* Real-time graphing or data logging capabilities must be active to monitor the setpoint, process variable, and control effort ($u(t)$).
* Physical safety stops and software limits must be engaged.

---


Phase 1 is the most critical step in a dual-loop setup because it bridges the gap between pure math and messy physical reality. Mathematical models assume that an actuator responds linearly to any control effort ($u(t) > 0$). In reality, mechanical systems suffer from stiction (static friction), deadband (like gear backlash), and sensor noise.

If you guess the threshold ($e_{th}$) incorrectly, your controller will either switch to the fine-tuning mode too early (causing a sluggish approach) or too late (causing the coarse controller to violently overshoot the target).

Here is a detailed breakdown of how to establish that boundary practically.

### 1. Open-Loop Stiction Testing

Before writing any PID logic, you need to find the exact minimum control effort required to make the physical hardware move.

* **The Process:** Bypass the PID loop completely. Feed a direct control signal (like a raw PWM duty cycle or voltage) to the actuator, starting at zero.
* **The Execution:** Slowly ramp up the signal in tiny increments (1%, 2%, 3%...). You can log the sensor response using a serial plotter, MATLAB, or a Python script listening to the serial port.
* **The Result:** The exact control value where the actuator finally breaks static friction and begins to move is your baseline. The small-signal zone is specifically designed to handle the error region where the required control effort hovers around this minimum driving force.

### 2. Measuring the Sensor Noise Floor

Digital sensors in embedded systems are rarely perfectly stable. If you set your error threshold smaller than your sensor's inherent noise, the controller will frantically toggle between the large and small signal states even when the robot is sitting perfectly still.

* **The Process:** Keep the actuator completely stationary.
* **The Execution:** Read the raw sensor data (e.g., encoder ticks or IMU angles) over several seconds. Calculate the maximum variance or deviation ($\pm \Delta e$) from the true position.
* **The Result:** Your threshold ($e_{th}$) must be strictly larger than this noise floor to prevent state-toggling instability.

### 3. Defining the Threshold ($e_{th}$)

With the physical limits quantified, you can set the numerical boundary. The threshold is the physical distance from the setpoint where you want the robot to stop "slowing down" and begin "settling."

* **Rule of Thumb:** A standard starting point is to set $e_{th}$ just outside the mechanical deadband. For example, if your drivetrain gears have a combined 3 degrees of backlash, setting the threshold to 5 degrees ensures the large-signal controller isn't fighting mechanical slop.

### 4. Software Implementation

The transition logic must execute efficiently within your discrete time step ($\Delta t$). It usually looks like a simple conditional structure evaluated right before the PID equation is calculated:

```c
// Evaluate absolute error against the threshold
if (abs(error) > e_th) {
    // Large Signal Mode: Fast approach, heavy braking
    Kp = Kp_large;
    Ki = 0;           // Keep strictly zero to prevent windup
    Kd = Kd_large;
} else {
    // Small Signal Mode: Precision settling
    Kp = Kp_small;
    Ki = Ki_small;    // Slowly overcomes final stiction
    Kd = Kd_small;
}

// Compute standard discrete PID using active parameters...

```

By completing this phase methodically, you ensure that the handoff between the two parameter sets occurs in a physically stable region, making the actual tuning in Phases 2 and 3 significantly easier.


### Phase 1: Boundary Establishment

1. **Observe Mechanical Limits:** Command the actuator using manual open-loop control to observe the system's static friction (stiction) and mechanical deadband.
2. **Set the Threshold ($e_{th}$):** Define the error boundary exactly outside the range of the system's deadband and sensor noise floor.
3. **Configure the Logic:** Program the controller to use Large-Signal parameters when $|e(t)| > e_{th}$ and Small-Signal parameters when $|e(t)| \le e_{th}$.

---

### Phase 2: Large-Signal (Coarse) Tuning

**Goal:** Drive large errors down to the $e_{th}$ boundary as rapidly as possible without violent overshoot.

1. **Zero All Parameters:** Set all $K_p$, $K_i$, and $K_d$ values for both operating regions to zero.
2. **Disable Integral Accumulation:** Ensure the large-signal $K_i$ remains strictly at zero to prevent windup during long travel distances.
3. **Increase Proportional Gain ($K_p$):** Incrementally increase the large-signal $K_p$. Apply large step changes to the setpoint. Stop increasing $K_p$ when the system reaches the $e_{th}$ boundary quickly but begins to overshoot aggressively.
4. **Apply Derivative Braking ($K_d$):** Incrementally increase the large-signal $K_d$ to act as a dampener. This will decelerate the actuator as the error rapidly decreases toward the boundary.
5. **Verify Slew Phase:** Repeat large step changes. The actuator should move at high speed and brake sharply, crossing the $e_{th}$ boundary with minimal residual velocity.

---

### Phase 3: Small-Signal (Fine) Tuning

**Goal:** Eliminate steady-state error and stabilize the system at the exact setpoint once inside the $e_{th}$ boundary.

1. **Set Initial Proportional Gain ($K_p$):** Set the small-signal $K_p$ to a value significantly lower than the large-signal $K_p$ to avoid high-frequency micro-oscillations around the setpoint.
2. **Introduce Integral Gain ($K_i$):** Gradually increase the small-signal $K_i$. Apply small step changes that keep the system entirely within the small-signal zone. The $I$ term should slowly ramp up the control effort to overcome stiction and eliminate the final steady-state error.
3. **Configure Anti-Windup:** Implement an integral clamp. The maximum accumulated integral value should be capped at the minimum control effort required to overcome static friction.
4. **Tune Derivative Gain ($K_d$):** If the small-signal $K_p$ and $K_i$ cause slight ringing or jittering at the setpoint, introduce a very small amount of small-signal $K_d$ to dampen the final settling phase.

---

### Phase 4: Handoff and Transition Verification

1. **Test Full Range Steps:** Apply a large step change to the setpoint that forces the system to start in the large-signal zone and transition into the small-signal zone.
2. **Monitor the Control Output ($u(t)$):** Observe the control effort graph at the exact moment the error crosses $e_{th}$.
3. **Smooth the Transition:** If there is a violent spike or drop in control effort at the boundary crossing, adjust the large-signal $K_d$ or the small-signal $K_p$ to match the required physical torque at that specific speed.
4. **Clear Integral Accumulator:** Ensure the software explicitly resets the integral accumulator to zero whenever the error exits the small-signal zone and enters the large-signal zone.