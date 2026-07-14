#include <Arduino.h>
#include "MotorControl.h"
#include "Screen.h"
#include "RLControl.h"

// Balance setpoint in millimeters
float target_x = 0;
float target_y = 0;

void setup() {
  Serial.begin(2000000);

  screen_init();
  motor_init();

  home_motors();  // drive to the hardstop offsets, then define that as the origin
  go_home();      // settle at the level (zero) position
  delay(1000);

  rl_reset_state();  // start the velocity filter clean
}

void loop() {
  // Reads the touchscreen, runs the trained actor network, and commands the
  // steppers in step space. Internally gated to ~30 Hz to match the training
  // cadence, and handles ball-lost (levels the plate after 3 s) on its own.
  rl_balance(target_x, target_y);

  // Must be called as fast as possible to actually step the motors toward
  // whatever target rl_balance last commanded.
  motorA.run();
  motorB.run();
  motorC.run();
}
