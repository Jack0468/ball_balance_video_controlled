// ---------------------------------------------------------------------------
// BallBalancingBot.ino -- Jetson-remote-control deployment.
//
// Dedicated firmware for the Jetson AGX Orin comparison arm (Option (c) in
// docs/plans -- see the plan file referenced from AGENTS.md's Phase 6 for the
// full rationale). This is a SEPARATE deployment from
// firmware/stm32_ml_control_and_vision/BallBalancingBot/ (the on-device RL
// firmware, source of truth for the laptop/expert-pipeline arm), not a
// modification of it -- each arm gets its own firmware image, built from
// shared modules, so switching arms is "flash a known image", not "hand-edit
// a shared sketch". See that directory's BallBalancingBot.ino for the
// RL-on-device variant.
//
// Bidirectional serial. One USB CDC link carries both directions:
//
//   PC (Jetson) -> MCU :  "S,<stepA>,<stepB>,<stepC>"
//                 pre-computed step targets from control_net.py (a NumPy
//                 port of the SAME trained policy RLControl.cpp runs
//                 on-device in the other firmware), running on the Jetson at
//                 whatever cadence the camera actually delivers instead of a
//                 compile-time-fixed rate. Parsed by RemoteStepControl.cpp.
//
//   MCU -> PC :  "T,<seq>,<ms>,<touch_x>,<touch_y>,<valid>,<a>,<b>,<c>"
//                 resistive touchscreen ground truth + actual stepper
//                 positions. The Jetson's control_net.py needs the actual
//                 step positions fed back each cycle (`actual_steps`) --
//                 this is where they come from. Emitted by TouchProbe.cpp
//                 at 25 Hz.
//
// IMPORTANT: this firmware does NOT compile in SerialCoords.cpp/Screen.h's
// downlink parser. RemoteStepControl.cpp's "S,..." line reader is the ONLY
// thing reading Serial here, on purpose -- see TouchProbe.h for why running
// both parsers on one Serial stream in the same loop() is unsafe (they'd
// race for bytes, up to silently losing every step command). If you need the
// original vision-coordinate downlink back, use the other firmware
// directory, don't add SerialCoords.cpp to this one.
//
// The touchscreen is a SENSOR ONLY for evaluation purposes; it does not feed
// the controller (the controller is entirely on the Jetson in this
// deployment).
//
// Everything the MCU prints that is not a "T," record starts with '#', so the
// host parser can skip it unambiguously. Do not add bare Serial.println()s.
// ---------------------------------------------------------------------------

#include <Arduino.h>
#include "TouchProbe.h"      // uplink: touchscreen ground truth + actual steps
#include "MotorControl.h"
#include "RemoteStepControl.h"

// USB CDC ignores the baud value, but keep it matched to the host's
// SERIAL_BAUD so a hardware-UART build works without edits.
#define SERIAL_BAUD 2000000

#define HEARTBEAT_MS 2000  // '#' status line cadence, 0 = off

static unsigned long last_heartbeat = 0;

void setup() {
  Serial.begin(SERIAL_BAUD);

  // Steppers first: home_motors() spins until the hardstops are found and we
  // do not want serial traffic interleaved with that.
  motor_init();
  home_motors();
  go_home();

  touch_probe_init();          // configure touchscreen pins, reset the uplink
  remote_step_control_init();  // clear the step-target parser/fail-safe state

  Serial.println("# ready proto=1 uplink=T downlink=S (jetson-remote-control)");
}

void loop() {
  // 1. Step-target control. Internally non-blocking: polls Serial for a
  //    fresh "S,<a>,<b>,<c>" line, applies safety clamping/timeout, and
  //    drives the steppers when a new target arrives. See RemoteStepControl.h
  //    for the fail-safe (hold, then level) behaviour when the Jetson stalls.
  remote_step_control_update();

  // 2. Ground truth. Rate-gated to 25 Hz, one ADC read per call, never blocks.
  touch_probe_update();

  // 3. Step the motors. Must run every iteration -- AccelStepper generates its
  //    pulses here, so anything above that blocks costs step timing.
  motorA.run();
  motorB.run();
  motorC.run();

#if HEARTBEAT_MS
  unsigned long now = millis();
  if (now - last_heartbeat >= HEARTBEAT_MS) {
    last_heartbeat = now;
    if (Serial.availableForWrite() > 48) {
      Serial.print("# hb remote=1");
      Serial.println();
    }
  }
#endif
}
