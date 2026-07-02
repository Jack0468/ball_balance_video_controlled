#include "PIDControllers.h"

// PID Constants
const fixed_t kp = 0.8;
const fixed_t ki = 0.2;
const fixed_t kd = 0.09;

// Maximums
const fixed_t max_output = 83.5;
const fixed_t max_angle = 12.5;

// Fixed dt for 1000Hz loop
const fixed_t dt = 0.001;

// State variables (static so they persist across loop iterations)
static fixed_t error_prev_x = 0;
static fixed_t error_prev_y = 0;
static fixed_t integ_x = 0;
static fixed_t integ_y = 0;

void balance_controller(
    fixed_t current_x, 
    fixed_t current_y, 
    fixed_t setpoint_x, 
    fixed_t setpoint_y, 
    fixed_t &out_thetaA, 
    fixed_t &out_thetaB, 
    fixed_t &out_thetaC
) {
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL
#pragma HLS INTERFACE s_axilite port=current_x bundle=CTRL
#pragma HLS INTERFACE s_axilite port=current_y bundle=CTRL
#pragma HLS INTERFACE s_axilite port=setpoint_x bundle=CTRL
#pragma HLS INTERFACE s_axilite port=setpoint_y bundle=CTRL
#pragma HLS INTERFACE s_axilite port=out_thetaA bundle=CTRL
#pragma HLS INTERFACE s_axilite port=out_thetaB bundle=CTRL
#pragma HLS INTERFACE s_axilite port=out_thetaC bundle=CTRL

    // X PID
    fixed_t error_x = current_x - setpoint_x;
    integ_x += error_x * dt;
    if (integ_x > (fixed_t)50) integ_x = 50;
    if (integ_x < (fixed_t)-50) integ_x = -50;
    fixed_t deriv_x = (error_x - error_prev_x) / dt;
    error_prev_x = error_x;

    fixed_t output_x = kp * error_x + ki * integ_x + kd * deriv_x;

    // Y PID
    fixed_t error_y = current_y - setpoint_y;
    integ_y += error_y * dt;
    if (integ_y > (fixed_t)50) integ_y = 50;
    if (integ_y < (fixed_t)-50) integ_y = -50;
    fixed_t deriv_y = (error_y - error_prev_y) / dt;
    error_prev_y = error_y;

    fixed_t output_y = kp * error_y + ki * integ_y + kd * deriv_y;

    // Constrain and map to angles
    if (output_x > max_output) output_x = max_output;
    if (output_x < -max_output) output_x = -max_output;
    if (output_y > max_output) output_y = max_output;
    if (output_y < -max_output) output_y = -max_output;

    fixed_t angle_x = output_x * (max_angle / max_output);
    fixed_t angle_y = output_y * (max_angle / max_output);

    // Call Inverse Kinematics
    // We arbitrarily set platform height h=120 for the balancing state
    CalculatedAngles angles = get_angles(angle_y, -angle_x, (fixed_t)120.0);

    out_thetaA = angles.thetaA;
    out_thetaB = angles.thetaB;
    out_thetaC = angles.thetaC;
}
