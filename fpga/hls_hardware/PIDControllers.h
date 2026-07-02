#ifndef PIDCONTROLLERS_H
#define PIDCONTROLLERS_H

#include "InverseKinematics.h"

// Top-level HLS function wrapper
void balance_controller(
    fixed_t current_x, 
    fixed_t current_y, 
    fixed_t setpoint_x, 
    fixed_t setpoint_y, 
    fixed_t &out_thetaA, 
    fixed_t &out_thetaB, 
    fixed_t &out_thetaC
);

#endif
