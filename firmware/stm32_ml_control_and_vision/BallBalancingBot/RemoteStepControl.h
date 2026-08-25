#ifndef REMOTESTEPCONTROL_H
#define REMOTESTEPCONTROL_H

// ---------------------------------------------------------------------------
// RemoteStepControl.h / .cpp
//
// Alternative control input: accepts PRE-COMPUTED motor step targets directly
// from an external host (e.g. the control net ported to the Jetson --
// host_software/ml_jetson_vla/core/control_net.py) and drives the steppers
// straight to them via MotorControl.cpp -- bypassing RLControl.cpp's
// on-device inference entirely.
//
// Use this INSTEAD OF rl_balance() when the "brain" deciding targets runs
// off-board. Do not call both in the same loop() -- they'd fight over pos[]
// and the steppers.
//
// WIRE PROTOCOL (ASCII, one sample per line, '\n' or '\r\n' terminated):
//
//     "S,<stepA>,<stepB>,<stepC>"     e.g.  "S,12,-40,7"
//
// Deliberately distinct from SerialCoords.cpp's bare-numeric ball-position
// format (leading 'S' + comma) so the two can never be confused if both ever
// end up wired to the same UART. Step targets are clamped to
// +/-REMOTE_STEP_LIMIT (see .cpp) before being applied -- the sender's own
// control net already bounds its output (see MAX_MOTOR_STEP in
// control_net.py), but this firmware does not trust that blindly, matching
// SerialCoords.cpp's own COORD_CLAMP precedent.
// ---------------------------------------------------------------------------

void remote_step_control_init();

// Call every loop iteration. Internally non-blocking; polls the serial
// buffer, applies safety clamping/timeout, and drives the steppers when a
// fresh target arrives. Mirrors rl_balance()'s call contract so swapping
// between the two in BallBalancingBot.ino is a one-line change.
void remote_step_control_update();

// True once per received, in-range sample. Exposed for telemetry/debugging;
// remote_step_control_update() already calls this internally.
bool remote_step_target_available();

#endif  // REMOTESTEPCONTROL_H
