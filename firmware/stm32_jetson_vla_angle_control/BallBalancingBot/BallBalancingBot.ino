// ---------------------------------------------------------------------------
// BallBalancingBot.ino -- Jetson-VLA-angle-control deployment.
//
// Dedicated firmware for the Arm 2 large-VLA comparison arm
// (host_software/ml_jetson_vla/docs/ARM2_MINIMAL_BASELINE_SCOPE.md): a large
// Qwen2.5-VL-3B model runs standalone on the Jetson AGX Orin, is prompted to
// emit a structured text output, and a host-side parser (not built by this
// task) turns that text into per-motor leg angles (theta_a/b/c, degrees).
// This firmware is the endpoint those angles are sent to.
//
// THIS IS A NEW, SEPARATE DIRECTORY, NOT A MODIFICATION OF EITHER EXISTING
// FIRMWARE:
//   - firmware/stm32_ml_control_and_vision/BallBalancingBot/ -- the on-device
//     RL firmware (source of truth for the laptop/expert-pipeline arm, Arm 1).
//     Untouched by this task.
//   - firmware/stm32_jetson_remote_control/BallBalancingBot/ -- accepts
//     PRE-COMPUTED STEP targets ("S,<a>,<b>,<c>") from control_net.py, a
//     NumPy port of Arm 1's trained RL policy running on the Jetson. This was
//     the closer starting point for THIS directory (copied from here, see
//     git history) because it already has the right shape -- external host
//     drives the steppers directly, no on-board inference, touchscreen
//     telemetry uplink -- but it is still Arm-1-specific (it exists to let
//     Arm 1's own control net run off-board) and its wire format is raw
//     step counts scaled by that specific policy's action space, not
//     degrees from an arbitrary external source. Untouched by this task.
// This directory exists because Arm 2's angle-shaped output has nowhere else
// to go: see AngleStepControl.h for the full rationale, wire protocol, and
// safety-bound derivation.
//
// Bidirectional serial. One USB CDC link carries both directions:
//
//   PC (Jetson) -> MCU :  "A,<theta_a>,<theta_b>,<theta_c>"
//                 per-motor leg angles in DEGREES, relative to each motor's
//                 homed zero -- the parsed output of the large VLM's prompted
//                 text response. Parsed by AngleStepControl.cpp, which
//                 converts to steps via MotorControl.cpp's angle_to_steps()
//                 and rejects (does not clamp-and-apply) anything outside
//                 the documented safety bound.
//
//   MCU -> PC :  "T,<seq>,<ms>,<touch_x>,<touch_y>,<valid>,<a>,<b>,<c>"
//                 resistive touchscreen ground truth + actual stepper
//                 positions, for evaluation telemetry. Identical format to
//                 the stm32_jetson_remote_control/ sibling deployment
//                 (TouchProbe.cpp here is an unmodified copy of that one).
//                 Emitted at 25 Hz.
//
// IMPORTANT: this firmware does NOT compile in SerialCoords.cpp/Screen.h's
// downlink parser or RemoteStepControl.cpp/.h. AngleStepControl.cpp's "A,..."
// line reader is the ONLY thing reading Serial here, on purpose -- see
// TouchProbe.h for why running more than one downlink parser on one Serial
// stream in the same loop() is unsafe (they'd race for bytes, up to silently
// losing every command). If you need the vision-coordinate downlink or the
// raw-step downlink back, use the other firmware directories, don't add
// their parsers to this one.
//
// The touchscreen is a SENSOR ONLY for evaluation purposes; it does not feed
// the controller (the controller -- the VLM's prompt/image loop -- is
// entirely on the Jetson in this deployment).
//
// Everything the MCU prints that is not a "T," record starts with '#', so the
// host parser can skip it unambiguously. Do not add bare Serial.println()s.
// ---------------------------------------------------------------------------

#include <Arduino.h>
#include "TouchProbe.h"      // uplink: touchscreen ground truth + actual steps
#include "MotorControl.h"
#include "AngleStepControl.h"

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
  angle_step_control_init();   // clear the angle-target parser/fail-safe state

  Serial.println("# ready proto=1 uplink=T downlink=A (jetson-vla-angle-control)");
}

void loop() {
  // 1. Angle-target control. Internally non-blocking: polls Serial for a
  //    fresh "A,<theta_a>,<theta_b>,<theta_c>" line, validates/converts it,
  //    and drives the steppers when a new in-range target arrives. See
  //    AngleStepControl.h for the fail-safe (hold, then level) behaviour
  //    when the Jetson/VLM stalls -- tuned for multi-second VLM inference
  //    cycles, not the ~30 Hz cadence the other two firmware variants
  //    assume.
  angle_step_control_update();

  // 2. Ground truth. Rate-gated to 25 Hz, one ADC read per call, never blocks.
  touch_probe_update();

  // 3. Step the motors. Must run every iteration -- AccelStepper generates its
  //    pulses here, so anything above that blocks costs step timing. This is
  //    also what keeps the plate actively holding its last commanded tilt
  //    between angle updates (see AngleStepControl.cpp's hold-timeout
  //    comment) -- distanceToGo() reaches 0 and run() becomes a cheap no-op,
  //    but the driver stays enabled (ENA held LOW since motor_init()) so the
  //    motors keep holding torque at position rather than going slack.
  motorA.run();
  motorB.run();
  motorC.run();

#if HEARTBEAT_MS
  unsigned long now = millis();
  if (now - last_heartbeat >= HEARTBEAT_MS) {
    last_heartbeat = now;
    if (Serial.availableForWrite() > 48) {
      Serial.print("# hb angle=1");
      Serial.println();
    }
  }
#endif
}
