#include "MotorControl.h"

#define STEPS_TO_ORIGIN_A 475 // steps offset from hardstop
#define STEPS_TO_ORIGIN_B 425 // steps offset from hardstop
#define STEPS_TO_ORIGIN_C 505 // steps offset from hardstop
#define ENA PD0 // ENA pin
#define h0 87 // height of platform when motors are at zero position
#define ks 100.0 // a constant to change our proportional speed function

// setup for steppers
AccelStepper motorA(1, PD3, PD2);  //(driver type, STEP, DIR) Driver A
AccelStepper motorB(1, PD5, PD4);  //(driver type, STEP, DIR) Driver B
AccelStepper motorC(1, PD7, PD6);  //(driver type, STEP, DIR) Driver C
long int pos[3]; // stores positions of stepper motors

//initializes motors by adding them in multistepper class
void motor_init() {
  motorA.setMaxSpeed(6000);
  motorB.setMaxSpeed(6000);
  motorC.setMaxSpeed(6000);

  motorA.setAcceleration(25000);
  motorB.setAcceleration(25000);
  motorC.setAcceleration(25000);
  
  // Since you physically flipped A and B to reverse them, we need to invert C in software
  // so it moves in the same direction!
  motorC.setPinsInverted(false, true, false);
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