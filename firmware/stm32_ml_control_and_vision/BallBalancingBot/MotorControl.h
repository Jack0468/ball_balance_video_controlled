#ifndef ROBOT_H
#define ROBOT_H

#include "InverseKinematics.h"
#include <Arduino.h>
#include <math.h>
#include "Screen.h"
#include <AccelStepper.h>

// Global stepper motor objects
extern AccelStepper motorA;
extern AccelStepper motorB;
extern AccelStepper motorC;
extern long int pos[3];


// Function prototypes
void motor_init();
long int angle_to_steps(double angle);
double steps_to_angle(int steps);
void home_motors();
void go_home();
void move_to_angle(double theta_deg, double phi_deg, double h);
void speed_controller();
void test_motor_speed();

#endif