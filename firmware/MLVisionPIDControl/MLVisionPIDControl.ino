#include <Arduino.h>
#include "MotorControl.h"
#include "Screen.h"
#include "PIDControllers.h"

void setup() {
  Serial.begin(2000000); // Fast baud rate for binary transfer & serial coordinates
  
  // We disable binary telemetry by default so we don't spam the Python host,
  // since the host doesn't actively read and parse it in the main PID script.
  enable_binary_telemetry = false;

  screen_init();
  motor_init();

  home_motors();
  go_home();
  delay(1000);
}

void loop() {
  // 1. Read target coordinates arriving over serial from host_software
  coords p = get_coords();
  
  // 2. The PID controller handles its own 30Hz (33ms) timer internally.
  // We pass it the target from the Host PC (p.target_x_mm, p.target_y_mm).
  pid_balance(p.target_x_mm, p.target_y_mm);
  
  // 3. Keep the serial coordinate buffer drained
  serial_coords_poll();
  
  // 4. Must be called as fast as possible to actually step the motors
  motorA.run();
  motorB.run();
  motorC.run();
}
