#include <Arduino.h>
#include "InverseKinematics.h"
#include "MotorControl.h"
#include "Screen.h"
#include "PIDControllers.h"
#include "DataCollectionStateMachine.h"

double target_x = 0;
double target_y = 0;

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
  // 1. Data Collection State Machine
  bool is_done = false;
  state_machine.getNextTarget(target_x, target_y, is_done);

  // If data collection is completely finished, dump the ball and lock motors
  if (is_done) {
    target_x = 0;
    target_y = 0;
    move_to_angle(12.0, 0, 80); // 12-degree pitch to roll ball off
    while(true) {
        // Lock infinitely
    }
  }
  
  // 3. The PID controller handles its own 30Hz (33ms) timer internally.
  // It will also automatically send the binary telemetry packet when it runs.
  pid_balance(0, 0);
  
  // 4. Must be called as fast as possible to actually step the motors
  motorA.run();
  motorB.run();
  motorC.run();
}