// ---------------------------------------------------------------------------
// BallBalancingBot.ino
//
// Bidirectional serial. One USB CDC link carries both directions:
//
//   PC  -> MCU :  "V,<seq>,<ball_x>,<ball_y>,<target_x>,<target_y>"
//                 vision estimate + audio-derived target. Drives the RL policy.
//                 Parsed by SerialCoords.cpp. UNCHANGED control path.
//
//   MCU -> PC :  "T,<seq>,<ms>,<touch_x>,<touch_y>,<valid>,<vis_x>,<vis_y>,<a>,<b>,<c>"
//                 resistive touchscreen ground truth + the vision frame it was
//                 taken against + actual stepper positions.
//                 Emitted by TouchProbe.cpp at 60 Hz.
//
// The touchscreen is a SENSOR ONLY. It never feeds the controller. That keeps
// the evaluation honest -- the policy sees exactly what it saw before this
// change, and the touchscreen is an independent witness to where the ball
// actually went.
//
// Everything the MCU prints that is not a "T," record starts with '#', so the
// host parser can skip it unambiguously. Do not add bare Serial.println()s.
//
// BUILD NOTE: the sketch folder must contain exactly one definition of
// screen_init()/get_coords()/mapf(). Delete or rename any leftover
// Screen.cpp / Screen-*.cpp backups -- the linker will otherwise fail with
// duplicate symbols. SerialCoords.cpp owns those now.
// ---------------------------------------------------------------------------

#include <Arduino.h>
#include "Screen.h"          // downlink: vision coords + target (SerialCoords.cpp)
#include "TouchProbe.h"      // uplink:   touchscreen ground truth
#include "MotorControl.h"
#include "RLControl.h"

// USB CDC ignores the baud value, but keep it matched to main.py's SERIAL_BAUD
// so a hardware-UART build works without edits.
#define SERIAL_BAUD 2000000

// Guard against running the plate with no host attached.
#define REQUIRE_HOST      0        // 1 = wait for the first vision frame before
                                   //     enabling the policy
#define HEARTBEAT_MS      2000     // '#' status line cadence, 0 = off

static unsigned long last_heartbeat = 0;

void setup() {
  Serial.begin(SERIAL_BAUD);

  // Steppers first: home_motors() spins until the hardstops are found and we do
  // not want serial traffic interleaved with that.
  motor_init();
  home_motors();
  go_home();

  screen_init();        // reset the downlink parser
  touch_probe_init();   // configure touchscreen pins, reset the uplink
  rl_reset_state();     // clear the velocity filter

  Serial.println("# ready proto=1 uplink=T downlink=V");
}

void loop() {
  // 1. Drain the RX buffer. Cheap, and doing it first keeps latency minimal.
  //    Note: do NOT call coords_available() here -- rl_balance() consumes that
  //    edge flag to pace itself, and calling it twice would eat the sample.
  serial_coords_poll();

  // 2. Target comes from the same downlink line as the ball position, so the
  //    policy sees a consistent (ball, target) pair from one vision frame.
  coords c = get_coords();

#if REQUIRE_HOST
  static bool host_seen = false;
  if (!host_seen) {
    if (serial_coords_seq() == 0) { touch_probe_update(); return; }
    host_seen = true;
  }
#endif

  // 3. Policy. Internally gated to the vision cadence; returns fast otherwise.
  rl_balance((float)c.target_x_mm, (float)c.target_y_mm);

  // 4. Ground truth. Rate-gated to 60 Hz, one ADC read per call, never blocks.
  touch_probe_update();

  // 5. Step the motors. Must run every iteration -- AccelStepper generates its
  //    pulses here, so anything above that blocks costs step timing.
  motorA.run();
  motorB.run();
  motorC.run();

#if HEARTBEAT_MS
  unsigned long now = millis();
  if (now - last_heartbeat >= HEARTBEAT_MS) {
    last_heartbeat = now;
    if (Serial.availableForWrite() > 48) {
      Serial.print("# hb seq=");
      // Explicit cast: uint32_t is 'unsigned int' on some cores and
      // 'unsigned long' on others, which makes print() ambiguous.
      Serial.print((unsigned long)serial_coords_seq());
      Serial.print(" ball=");
      Serial.println(c.z > 0 ? 1 : 0);
    }
  }
#endif
}