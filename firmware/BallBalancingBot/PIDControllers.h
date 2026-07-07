#ifndef PIDCONTROLLERS_H
#define PIDCONTROLLERS_H

#include "MotorControl.h"
#include "InverseKinematics.h"
#include <Arduino.h>
#include <math.h>
#include "Screen.h"
#include <AccelStepper.h>
#include <MultiStepper.h>

void pid_balance_with_coords(double setpoint_x, double setpoint_y, double current_x, double current_y);
void move_to_point(double setpoint_x, double setpoint_y, unsigned long delay);
void move_line(double rx, double ry, double speed, int repeat = 0);
void move_ellipse(double rx, double ry, double speed, double repeat = 0);
void move_square(double side_length, double speed, int repeat = 0);
void move_figure8(double radius, double speed, int repeat = 0);
void move_spiral(double max_radius, double speed, int repeat = 0);
void move_star(double radius, double speed, int repeat = 0);
void move_heart(double size, double speed, int repeat = 0);
void muteAllSerialOutput();
void enableAllSerialOutput();

extern double error[2];
extern double integ[2];
extern double deriv[2];
extern double output_angles[2];

#endif