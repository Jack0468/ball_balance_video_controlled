# Classical Control

This directory contains classical control algorithms and mathematics required for the physical stabilization of the robot. 
These modules are used in tandem with, or as baselines for, the Machine Learning control policies.

## Modules

- **`pid_controller.py`**: Implements standard Proportional-Integral-Derivative (PID) control loops for stabilizing the ball at the center (or targeted offset) of the platform.
- **`inverse_kinematics.py`**: Mathematical derivations and translation matrices that convert a desired 3D platform tilt (pitch/roll) into precise linear actuator positions for the stepper motors.
