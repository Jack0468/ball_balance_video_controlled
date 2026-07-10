#include <Arduino.h>
#include "InverseKinematics.h"
#include "MotorControl.h"
#include "Screen.h"
#include "PIDControllers.h"
#include "DataCollectionStateMachine.h"

DataCollectionStateMachine state_machine;
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
  static bool finished = false;
  
  if (finished) {
    // Do nothing! The motors will just hold their final tilted position, locking the platform.
    return;
  }

  bool is_done = false;
  state_machine.getNextTarget(target_x, target_y, is_done);
  
  if (is_done) {
    // Tip the ball completely off the plane (12 degrees pitch)
    move_to_angle(12.0, 0.0, 80);
    
    // Blocking loop to ensure the motors actually execute the tilt before we freeze the program
    while (motorA.distanceToGo() != 0 || motorB.distanceToGo() != 0 || motorC.distanceToGo() != 0) {
      motorA.run();
      motorB.run();
      motorC.run();
    }
    
    finished = true;
    return;
  }
  
  // The PID controller handles its own 30Hz (33ms) timer internally.
  // It will also automatically send the binary telemetry packet when it runs.
  pid_balance(target_x, target_y);
  
  // Must be called as fast as possible to actually step the motors
  motorA.run();
  motorB.run();
  motorC.run();
}
