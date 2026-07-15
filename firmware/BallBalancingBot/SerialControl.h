#ifndef SERIAL_CONTROL_H
#define SERIAL_CONTROL_H

#include <Arduino.h>

void check_serial_commands(double &cam_x, double &cam_y, bool &cam_active);

#endif
