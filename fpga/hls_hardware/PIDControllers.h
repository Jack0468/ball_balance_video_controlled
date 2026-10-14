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
    fixed_t &out_thetaC,
    // true if any leg's IK target was unreachable this call (see
    // InverseKinematics.h CalculatedAngles::fault) -- previously unreported;
    // added 2026-10-14 per docs/TODO.md item 600.
    bool &out_fault
);

#endif
