#ifndef TELEMETRY_H
#define TELEMETRY_H

// ---------------------------------------------------------------------------
// Telemetry.h / .cpp  --  PROPOSAL, staged for review, not yet merged into
// firmware/stm32_ml_control_and_vision/BallBalancingBot/. See
// ../TELEMETRY_PROTOCOL.md for the full rationale and wire format.
//
// NOT designed from scratch -- firmware/MLVisionPIDControl/PIDControllers.cpp
// (found 2026-08-18, a PID-based firmware variant not in AGENTS.md's list of
// known BallBalancingBot.ino duplicates -- it's differently named, so a
// filename-based audit wouldn't have caught it) already has a real, working
// binary telemetry implementation: a packed TelemetryPacket struct + the same
// 0xAABBCCDD-style sync-header trick, gated by an `enable_binary_telemetry`
// flag, sent from inside pid_balance(). This proposal reuses that same
// pattern (packed struct, sync header as the struct's own first field,
// micros() not millis()) rather than inventing a divergent one, so it looks
// like "the same mechanism you already use elsewhere" to whoever reviews it.
// Field *set* differs because the control law differs: PIDControllers.cpp's
// struct carries PID-specific diagnostics (error/integral/deriv terms) that
// don't exist under RLControl.cpp's control law, and adds actual_step_a/b/c
// (see below) which the PID variant has no use for.
//
// Standalone outbound telemetry, independent of which control path is active
// (rl_balance() / SerialCoords.cpp today, or a future remote_step_control_
// update() / RemoteStepControl.cpp Phase-B path) -- call telemetry_send()
// once per loop() iteration after the active control path has run, passing
// whatever touch/target mm values that path has available. Under the current
// rl_balance() path these come from get_coords() (Screen.h); a future
// RemoteStepControl-only path has no mm-space touch/target at all (it only
// ever sees step targets), so pass 0.0f for both in that case -- the step/
// angle fields are still meaningful either way since they read directly off
// the steppers, not off whichever coordinate path is active.
//
// Deliberately reads motorA/B/C.currentPosition() (real, lagged stepper
// position), NOT the pos[] target-buffer array (MotorControl.h) -- confirmed
// by reading RLControl.cpp directly that this is what the control net itself
// treats as "actual_steps". Telemetry must match that source or Phase B's
// closed loop (control_net.py on the Jetson) runs on the wrong signal.
//
// FLAG FOR THE FIRMWARE OWNER, not silently resolved here: PIDControllers.cpp's
// own telemetry instead logs steps_to_angle(pos[i]) -- the commanded-target
// buffer, not currentPosition(). That's a real inconsistency between the two
// firmware variants, not a typo on one side necessarily -- it may be an
// intentional/acceptable approximation under PID (where pos[] tracks the
// setpoint closely at 30Hz) that just doesn't hold for the RL control net's
// stated training assumptions. Flagging rather than assuming either
// convention is "the bug."
// ---------------------------------------------------------------------------

void telemetry_init();

// touch_x_mm/touch_y_mm/target_x_mm/target_y_mm: whatever the calling control
// path currently has (0.0f if not applicable, e.g. a future step-only path).
void telemetry_send(float touch_x_mm, float touch_y_mm, float target_x_mm, float target_y_mm);

#endif  // TELEMETRY_H
