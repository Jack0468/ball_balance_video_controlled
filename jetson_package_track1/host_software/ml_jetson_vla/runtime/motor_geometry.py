"""Python port of `steps_to_angle()`/`angle_to_steps()` from
`firmware/stm32_ml_control_and_vision/BallBalancingBot/MotorControl.cpp` (read-only
reference for this module -- that firmware file is not modified here or anywhere else
in `ml_jetson_vla`).

Confirmed by reading the real source (2026-09-15), not guessed from the geometry:

    #define STEPS_TO_ORIGIN_A 175 // steps offset from hardstop
    #define STEPS_TO_ORIGIN_B 190
    #define STEPS_TO_ORIGIN_C 185

    long int angle_to_steps(double angle) { return round((3200.0 / 360.0) * angle); }
    double   steps_to_angle(int steps)    { return (360.0 / 3200.0) * steps; }

3200 steps/revolution, 360 degrees/revolution -- a plain linear map, no gear ratio or
non-linear leg geometry folded in here (that lives in `InverseKinematics.cpp`, upstream
of `angle_to_steps()`, and is out of scope for this port).

STEPS_TO_ORIGIN_A/B/C are NOT part of this conversion. They are consumed once, inside
`home_motors()`, to drive each motor to its physical hardstop-relative zero position;
immediately after, `home_motors()` calls `motorX.setCurrentPosition(0)`, which redefines
that homed position as step 0 in the AccelStepper library's own internal counter. Every
step count this project ever reads back afterwards -- including `motorA.currentPosition()`,
which is exactly what `TouchProbe.cpp` sends as `motor_a/b/c` in the `T,...` uplink
(`src/touch_logger.py`'s `motor_a`/`motor_b`/`motor_c` CSV columns) -- is already relative
to that homed zero. So `steps_to_angle()` applies directly to live `motor_a/b/c` telemetry
with no origin-offset subtraction needed; re-applying STEPS_TO_ORIGIN_* here would double
count an offset the firmware has already zeroed out.

This module is used ONLY to convert telemetry into the `theta_a/b/c` training schema
(`ml_multimodal/data_processing/generate_vla_dataset.py`, `ml_multimodal/core/dataset.py`,
`docs/ARCHITECTURE.md`'s standard action representation). It does not touch, and must
never be imported into, any control path -- the STM32 remains the sole owner of real
step/angle conversion for actuation.
"""

from __future__ import annotations

STEPS_PER_REV = 3200.0
DEG_PER_REV = 360.0
DEG_PER_STEP = DEG_PER_REV / STEPS_PER_REV  # 0.1125 deg/step


def steps_to_angle(steps: float) -> float:
    """Port of `MotorControl.cpp::steps_to_angle()`. `steps` is a raw AccelStepper step
    count relative to the homed zero position (e.g. `motor_a/b/c` from the `T,...`
    telemetry uplink) -- NOT a hardstop-relative count, see module docstring."""
    return DEG_PER_STEP * steps


def angle_to_steps(angle_deg: float) -> int:
    """Port of `MotorControl.cpp::angle_to_steps()`, provided for completeness/symmetry
    (e.g. a future Arm 3 FPGA bridge converting a large-VLA model's `theta_a/b/c` output
    back to steps, per `docs/ARCHITECTURE.md`'s note on `angle_to_steps()`). Not used by
    the Track 4 telemetry-to-training-schema conversion this module was built for, which
    only needs the steps->angle direction. Matches the firmware's `round()` (round-half-
    away-from-zero); Python's built-in `round()` is round-half-to-even, so this uses an
    explicit away-from-zero rounding to match the C++ behavior exactly at the .5 boundary."""
    scaled = angle_deg / DEG_PER_STEP
    if scaled >= 0:
        return int(scaled + 0.5)
    return -int(-scaled + 0.5)
