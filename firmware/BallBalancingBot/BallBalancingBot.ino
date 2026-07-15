#include <Arduino.h>
#include "InverseKinematics.h"
#include "MotorControl.h"
#include "Screen.h"
#include "PIDControllers.h"
#include "SerialControl.h"
#include "DataCollectionStateMachine.h"

double target_x = 0;
double target_y = 0;
double cam_x = 0;
double cam_y = 0;
bool cam_active = false;

DataCollectionStateMachine state_machine;

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
  // 1. Read Serial for coordinates (updates cam_x, cam_y, cam_active)
  check_serial_commands(cam_x, cam_y, cam_active);
  
  // 2. Generate target using the state machine
  bool is_done = false;
  state_machine.getNextTarget(target_x, target_y, is_done);

  // If data collection is done, lock the motors in a dump position
  if (is_done) {
    move_to_angle(12, 0, 80);
    motorA.run();
    motorB.run();
    motorC.run();
    return;
  }
  
  // 3. The PID controller handles its own 30Hz (33ms) timer internally.
  // It will also automatically send the binary telemetry packet when it runs.
  pid_balance(target_x, target_y, cam_x, cam_y, cam_active);
  
  // 4. Must be called as fast as possible to actually step the motors
  motorA.run();
  motorB.run();
  motorC.run();
}
