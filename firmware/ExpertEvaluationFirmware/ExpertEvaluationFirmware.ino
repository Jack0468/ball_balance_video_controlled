#include <Arduino.h>
#include "MotorControl.h"
#include "Screen.h"
#include "RLControl.h"
#include "TouchProbe.h"



void setup() {
  Serial.begin(2000000);

  screen_init();
  motor_init();

  home_motors();  // drive to the hardstop offsets, then define that as the origin
  go_home();      // settle at the level (zero) position
  delay(1000);

  touch_probe_init();
  rl_reset_state();  // start the velocity filter clean
}

void loop() {
  // Reads the latest host-provided vision coordinates and runs the trained
  // actor network. The touchscreen logger also samples and transmits its
  // ground-truth record in parallel.

  coords p = get_coords();
  rl_balance(p.target_x_mm, p.target_y_mm);

  serial_coords_poll();
  touch_probe_update();

  // Must be called as fast as possible to actually step the motors toward
  // whatever target rl_balance last commanded.
  motorA.run();
  motorB.run();
  motorC.run();
}
