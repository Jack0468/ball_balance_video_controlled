#include <Arduino.h>
#include "MotorControl.h"
#include "Screen.h"
#include "RLControl.h"



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

  coords p = get_coords();
  rl_balance(p.target_x_mm, p.target_y_mm);

  serial_coords_poll();  

  // Must be called as fast as possible to actually step the motors toward
  // whatever target rl_balance last commanded.
  motorA.run();
  motorB.run();
  motorC.run();
}
