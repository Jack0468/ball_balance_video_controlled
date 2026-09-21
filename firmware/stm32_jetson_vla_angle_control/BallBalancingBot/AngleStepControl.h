#ifndef ANGLESTEPCONTROL_H
#define ANGLESTEPCONTROL_H

// ---------------------------------------------------------------------------
// AngleStepControl.h / .cpp
//
// WHY THIS FIRMWARE DIRECTORY EXISTS (firmware/stm32_jetson_vla_angle_control/):
// Arm 2 of the VLA comparison (host_software/ml_jetson_vla/docs/ARM2_MINIMAL_BASELINE_SCOPE.md)
// is a large Qwen2.5-VL-3B vision-language model running standalone on the
// Jetson AGX Orin. It has no native action head -- it is prompted to emit a
// structured TEXT output that a host-side parser turns into per-motor leg
// angles (theta_a/b/c, degrees), not step counts, not (x,y) ball/target
// positions. Neither existing firmware accepts that:
//   - stm32_ml_control_and_vision/SerialCoords.cpp only accepts
//     (ball_x, ball_y, target_x, target_y) in mm for the on-board RL net.
//   - stm32_jetson_remote_control/RemoteStepControl.cpp accepts raw STEP
//     targets ("S,<a>,<b>,<c>"), specifically the step-space output of
//     control_net.py (a port of the trained RL policy) -- not degrees, and
//     not meant for an arbitrary external angle source.
// This module is a THIRD, deliberate variant: same "external host drives the
// steppers directly, no on-board inference" shape as RemoteStepControl.cpp
// (it was the closer starting point and is copied from there, not from
// SerialCoords.cpp), but the wire format and safety bound are redefined
// around ANGLES because that's what the Qwen-derived model's parsed output
// actually is. See BallBalancingBot.ino's header for why this lives in its
// own directory rather than editing either original in place (this project
// has a standing, explicitly-documented risk of firmware-directory
// duplication confusion -- see CLAUDE.md -- so this variant states its
// purpose here instead of leaving it implicit).
//
// USE THIS INSTEAD OF rl_balance() (on-board RL) AND INSTEAD OF
// remote_step_control_update() (raw step targets) -- do not call more than
// one control module in the same loop(), they'd fight over pos[] and the
// steppers.
//
// WIRE PROTOCOL (ASCII, one sample per line, '\n' or '\r\n' terminated):
//
//     "A,<theta_a>,<theta_b>,<theta_c>"     e.g.  "A,5.25,-3.10,0.00"
//
// theta_a/b/c are PER-MOTOR LEG ANGLES in DEGREES, relative to each motor's
// homed zero position (the "parallel to the ground" pose set by
// home_motors()/go_home() in MotorControl.cpp) -- exactly the quantity
// MotorControl.cpp's angle_to_steps()/steps_to_angle() already convert,
// and exactly the `theta_a/b/c` action representation
// host_software/ml_jetson_vla/runtime/motor_geometry.py documents as this
// project's standard action schema for a parsed large-VLA text output. This
// is NOT platform tilt (theta/phi fed to InverseKinematics.cpp's
// get_angles()) -- the VLM's parser is expected to emit the three
// already-per-leg angles directly, the same quantity RLControl.cpp's trained
// net implicitly targets via its step-space output.
//
// Distinct leading tag ('A' vs RemoteStepControl.cpp's 'S' vs
// SerialCoords.cpp's bare-numeric format) so none of the three downlink
// parsers can ever misinterpret another's line if wires ever get crossed.
//
// SAFETY BOUND -- read before changing MAX_MOTOR_ANGLE_DEG:
// There is no independently-documented mechanical hardstop travel range (in
// degrees) anywhere in this codebase as of 2026-09-18 -- home_motors() finds
// each hardstop via STEPS_TO_ORIGIN_A/B/C and defines that as the physical
// zero, but no file states how many further degrees of travel are safe past
// that zero. The one real, already-validated bound that exists is
// MAX_MOTOR_STEP = 98 steps, used identically in three places:
// RLControl.cpp (the trained on-board RL net's action-space clip),
// control_net.py (the Jetson port of that same net -- "do not change these
// without changing the firmware too" per its own comment), and
// RemoteStepControl.cpp's REMOTE_STEP_LIMIT (which reuses it explicitly as
// "a sane conservative default"). This module follows that same precedent:
// MAX_MOTOR_ANGLE_DEG below is 98 steps converted via MotorControl.cpp's own
// angle_to_steps() convention (3200 steps/rev, 360 deg/rev -> 0.1125 deg/step
// -> 98 * 0.1125 = 11.025 degrees), not an independently re-derived
// mechanical limit. Tighten it if a real hardstop spec is ever documented;
// do not widen it without that documentation.
//
// OUT-OF-RANGE POLICY -- deliberately different from RemoteStepControl.cpp:
// RemoteStepControl.cpp silently clamps an out-of-range component to the
// limit and still applies the (now-clamped) move. This module instead
// REJECTS the entire line (drops it, applies nothing, logs a "# ERR"
// line) if ANY of the three angles is out of range -- a clamped-but-silently
// -applied angle from an unvalidated text parser (the VLM's output, parsed
// by a script that has never run against this firmware before) is exactly
// the kind of input this project's own operating instructions call out as
// something to fail loudly on rather than paper over.
// ---------------------------------------------------------------------------

void angle_step_control_init();

// Call every loop iteration. Internally non-blocking; polls the serial
// buffer, validates/converts angle targets, and drives the steppers when a
// fresh in-range sample arrives. Mirrors rl_balance()'s /
// remote_step_control_update()'s call contract.
void angle_step_control_update();

// True once per received, in-range sample. Exposed for telemetry/debugging;
// angle_step_control_update() already calls this internally.
bool angle_step_target_available();

#endif  // ANGLESTEPCONTROL_H
