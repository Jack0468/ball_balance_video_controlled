#ifndef LEG_H
#define LEG_H

#include "MotorControl.h"
#include "InverseKinematics.h"
#include <Arduino.h>
#include <math.h>
#include "Screen.h"
#include <AccelStepper.h>
#include <MultiStepper.h>

void pid_balance(double setpoint_x, double setpoint_y);
void move_to_point(double setpoint_x, double setpoint_y, unsigned long delay);

// Exported for telemetry logging
extern bool enable_binary_telemetry;
extern double current_ball_x;
extern double current_ball_y;
extern bool ball_detected;
extern double error[2];
extern double integ[2];
extern double deriv[2];
extern double output_angles[2];

extern bool enable_binary_telemetry;

#endif