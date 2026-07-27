#include <Arduino.h>
#include <MotorControl.h> // From VRI_Core symlink!
#include "SerialControl.h"

// VLA Direct Motor Angles
double theta_a = 0.0;
double theta_b = 0.0;
double theta_c = 0.0;

void setup() {
    Serial.begin(115200);
    motor_init();
    go_home();
}

void loop() {
    // 1. Check Serial for new [theta_a, theta_b, theta_c] from Host VLA
    check_vla_serial_commands(theta_a, theta_b, theta_c);
    
    // 2. Drive motors directly to those angles!
    // Note: move_to_angle usually expects platform roll/pitch, 
    // but here the VLA calculates raw servo angles directly.
    // If VLA outputs theta directly, we bypass IK and just drive servos.
    
    // Convert degrees to steps
    long steps_a = angle_to_steps(theta_a);
    long steps_b = angle_to_steps(theta_b);
    long steps_c = angle_to_steps(theta_c);
    
    // Drive
    motorA.moveTo(steps_a);
    motorB.moveTo(steps_b);
    motorC.moveTo(steps_c);
    
    motorA.run();
    motorB.run();
    motorC.run();
}
