#include "MotorControl.h"

#define STEPS_TO_ORIGIN_A 175 // steps offset from hardstop
#define STEPS_TO_ORIGIN_B 190 // steps offset from hardstop
#define STEPS_TO_ORIGIN_C 185 // steps offset from hardstop
#define ENA PD0 // ENA pin
#define h0 87 // height of platform when motors are at zero position
#define ks 100.0 // a constant to change our proportional speed function

// setup for steppers
AccelStepper motorA(1, PD3, PD2);  //(driver type, STEP, DIR) Driver A
AccelStepper motorB(1, PD5, PD4);  //(driver type, STEP, DIR) Driver B
AccelStepper motorC(1, PD7, PD6);  //(driver type, STEP, DIR) Driver C
long int pos[3]; // stores positions of stepper motors
double speed_prev[3] = {0, 0, 0}; // stores previous speeds for smoothing

//initializes motors by adding them in multistepper class
void motor_init() {
  motorA.setMaxSpeed(6000);
  motorB.setMaxSpeed(6000);
  motorC.setMaxSpeed(6000);

  motorA.setAcceleration(25000);
  motorB.setAcceleration(25000);
  motorC.setAcceleration(25000);
  
  // Motor directions will be calibrated and fixed purely in hardware.
  // No software inversion is applied.
  pinMode(ENA, OUTPUT);
  digitalWrite(ENA, HIGH);
  delay(2000);
  digitalWrite(ENA, LOW);
}

// angle in degrees to the nearest step (rounds up)
long int angle_to_steps(double angle) {
  return round((3200.0 / 360.0) * angle);
}

//steps to nearest angle (as a decimal)
double steps_to_angle(int steps) {
  return (360.0 / 3200.0) * steps;
}

//moves motors to position where bottom leg is parallel to the ground. this is the zero position
void home_motors() {

  // Move NEGATIVE steps to go UP (away from hardstops)
  // Because the A/B motors were physically flipped, positive steps move DOWN!
  pos[0] = STEPS_TO_ORIGIN_A;
  pos[1] = STEPS_TO_ORIGIN_B;
  pos[2] = STEPS_TO_ORIGIN_C;

  //calculates position and moves all motors to the zero position
  motorA.moveTo(pos[0]);
  motorB.moveTo(pos[1]);
  motorC.moveTo(pos[2]);

  while (motorA.distanceToGo() != 0 || motorB.distanceToGo() != 0 || motorC.distanceToGo() != 0) {
    motorA.run();
    motorB.run();
    motorC.run();
  }

  //makes the new point the origin (zero position)
  motorA.setCurrentPosition(0);
  motorB.setCurrentPosition(0);
  motorC.setCurrentPosition(0);

  Serial.println("Motors to zero position");
}

//goes to zero position (home). requires home_motors first.
void go_home() {
  motorA.moveTo(0);
  motorB.moveTo(0);
  motorC.moveTo(0);

  while (motorA.distanceToGo() != 0 || motorB.distanceToGo() != 0 || motorC.distanceToGo() != 0) {
    motorA.run();
    motorB.run();
    motorC.run();
  }
}

// calculates positions to move to a specific angle, but only moves at most one step per call.
void move_to_angle(double theta_deg, double phi_deg, double h) {

  //uses get_angles function to get positions required to move to position
  CalculatedAngles result = get_angles(theta_deg, phi_deg, h);
  pos[0] = angle_to_steps(result.thetaA);
  pos[1] = angle_to_steps(result.thetaB);
  pos[2] = angle_to_steps(result.thetaC);

  motorA.moveTo(pos[0]);
  motorB.moveTo(pos[1]);
  motorC.moveTo(pos[2]);
}

// Calculates proportional motor speeds for all three motors to make movements smooth
void speed_controller() {
  double current_pos[3];

  current_pos[0] = motorA.currentPosition();
  current_pos[1] = motorB.currentPosition();
  current_pos[2] = motorC.currentPosition();

  for (int i = 0; i < 3; i++) {
    double target_speed = abs(current_pos[i] - pos[i]) * ks;
    // Constrain speed so there aren't any sudden jumps
    target_speed = constrain(target_speed, speed_prev[i] - 300, speed_prev[i] + 300);
    // Constrain to physical limits (min 10 so it doesn't get permanently stuck if error is tiny)
    target_speed = constrain(target_speed, 10, 1300);
    
    speed_prev[i] = target_speed;

    if (i == 0) motorA.setMaxSpeed(target_speed);
    if (i == 1) motorB.setMaxSpeed(target_speed);
    if (i == 2) motorC.setMaxSpeed(target_speed);
  }
}