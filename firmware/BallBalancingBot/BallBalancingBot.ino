#include <Arduino.h>
#include "InverseKinematics.h"
#include "MotorControl.h"
#include "Screen.h"
#include "PIDControllers.h"
#include "SerialControl.h"

double target_x = 0;
double target_y = 0;

void setup() {
  Serial.begin(2000000); // Fast baud rate for binary transfer
  
  // We want to stream binary data, so disable human-readable prints
  enable_binary_telemetry = true;

  screen_init();
  motor_init();

  home_motors();
  go_home();
  delay(1000);
}

void loop() {
  // 1. Read Serial for coordinates (with 100ms failsafe)
  check_serial_commands(target_x, target_y);
  
  // 2. The PID controller handles its own 30Hz (33ms) timer internally.
  // It will also automatically send the binary telemetry packet when it runs.
  pid_balance(target_x, target_y);
  
  // 3. Must be called as fast as possible to actually step the motors
  motorA.run();
  motorB.run();
  motorC.run();
}
