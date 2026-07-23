#include <math.h>
#include <Arduino.h>
#include "MotorControl.h"
#include "RLControl.h"
#include "weights_old1.h"

#define CONTROL_DT       0.0333f   // 30 Hz control period (seconds)
#define VEL_FILTER_ALPHA 0.35f
#define MAX_MOTOR_STEP   98.0f

extern coords get_coords();          // returns ball x_mm, y_mm, z(detected flag)
extern long int pos[3];              // MotorControl's target-position array
extern void speed_controller();      // MotorControl proportional-speed smoother

// Motor objects live in MotorControl.cpp
extern AccelStepper motorA;
extern AccelStepper motorB;
extern AccelStepper motorC;


static float prev_obs_pos[2] = {0.0f, 0.0f};
static float filt_vel[2]     = {0.0f, 0.0f};
static bool  have_prev_pos   = false;

void rl_reset_state() {
    prev_obs_pos[0] = prev_obs_pos[1] = 0.0f;
    filt_vel[0] = filt_vel[1] = 0.0f;
    have_prev_pos = false;
}

static void forward_pass(const float obs[NN_IN], float out[NN_OUT]) {
    float h1[NN_H1];
    for (int j = 0; j < NN_H1; j++) {
        float acc = NN_B1[j];
        for (int i = 0; i < NN_IN; i++) {
            acc += obs[i] * NN_W1[i * NN_H1 + j];   // W1[i][j]
        }
        h1[j] = tanhf(acc);
    }

    float h2[NN_H2];
    for (int j = 0; j < NN_H2; j++) {
        float acc = NN_B2[j];
        for (int i = 0; i < NN_H1; i++) {
            acc += h1[i] * NN_W2[i * NN_H2 + j];    // W2[i][j]
        }
        h2[j] = tanhf(acc);
    }

    for (int j = 0; j < NN_OUT; j++) {
        float acc = NN_B3[j];
        for (int i = 0; i < NN_H2; i++) {
            acc += h2[i] * NN_W3[i * NN_OUT + j];   // W3[i][j]
        }
        // clip to action space [-1, 1]
        if (acc >  1.0f) acc =  1.0f;
        if (acc < -1.0f) acc = -1.0f;
        out[j] = acc;
    }
}

// ---------------------------------------------------------------------------
// Build observation, run the net, return step targets. Pure function of its
// inputs plus the velocity filter state -- this is what the self-test calls.
// ---------------------------------------------------------------------------
void rl_infer(float x_mm, float y_mm,
              float target_x_mm, float target_y_mm,
              const float actual_steps[3],
              float target_steps_out[3],
              float actual_dt) {

    // velocity by finite difference of position, then EMA low-pass.
    // On the very first sample we have no previous position: report zero
    // velocity (matches the sim's cold-start, which self-corrects in a step).
    float raw_vx = 0.0f, raw_vy = 0.0f;
    if (have_prev_pos) {
        raw_vx = (x_mm - prev_obs_pos[0]) / actual_dt;
        raw_vy = (y_mm - prev_obs_pos[1]) / actual_dt;
    }
    filt_vel[0] = VEL_FILTER_ALPHA * raw_vx + (1.0f - VEL_FILTER_ALPHA) * filt_vel[0];
    filt_vel[1] = VEL_FILTER_ALPHA * raw_vy + (1.0f - VEL_FILTER_ALPHA) * filt_vel[1];
    prev_obs_pos[0] = x_mm;
    prev_obs_pos[1] = y_mm;
    have_prev_pos = true;

    // 9-element observation, RAW units (scaling folded into weights).
    float obs[NN_IN];
    obs[0] = x_mm;
    obs[1] = y_mm;
    obs[2] = x_mm - target_x_mm;
    obs[3] = y_mm - target_y_mm;
    obs[4] = filt_vel[0];
    obs[5] = filt_vel[1];
    obs[6] = actual_steps[0];
    obs[7] = actual_steps[1];
    obs[8] = actual_steps[2];

    float action[NN_OUT];
    forward_pass(obs, action);

    // action in [-1,1] -> integer step target, matching training's np.round.
    for (int j = 0; j < NN_OUT; j++) {
        target_steps_out[j] = roundf(action[j] * MAX_MOTOR_STEP);
    }
}

// ---------------------------------------------------------------------------
// Main control loop. Call every iteration; gated internally to 30 Hz.
// ---------------------------------------------------------------------------
void rl_balance(float target_x_mm, float target_y_mm) {
    static unsigned long t_prev = 0;
    static unsigned long last_detected_time = 0;

    bool fresh = coords_available();
    unsigned long t = millis();
    if (!fresh && (t - t_prev) < 100) return;

    // if (t_prev == 0 || dt > 0.5f) {   // first run or a big stall: resync timing
    //     t_prev = t;
    //     return;
    // }
    // if (dt < CONTROL_DT) return;      // hold cadence at ~30 Hz
    // t_prev = t;

    float dt = (t - t_prev) / 1000.0f;
    if (t_prev == 0 || dt > 0.5f) {   // first run or a big stall: resync timing
        t_prev = t;
        return;
    }
    if (dt < 0.005f) dt = 0.005f;     // floor: never divide velocity by ~0
    t_prev = t;


    coords p = get_coords();
    bool detected = (p.z > 0);

    if (!detected) {
        // Ball lost. After 3 s, level the plate and clear the velocity filter,
        // mirroring the PID version's safety behaviour.
        if (t - last_detected_time >= 3000) {
            rl_reset_state();
            motorA.moveTo(0);
            motorB.moveTo(0);
            motorC.moveTo(0);
            pos[0] = pos[1] = pos[2] = 0;
            speed_controller();
        }
        return;
    }
    last_detected_time = t;

    // Read ACTUAL stepper positions -- the network trained on lagged motor
    // state, so we must NOT feed it the last commanded target.
    float actual_steps[3] = {
        (float)motorA.currentPosition(),
        (float)motorB.currentPosition(),
        (float)motorC.currentPosition()
    };

    float target_steps[3];
    rl_infer(p.x_mm, p.y_mm, target_x_mm, target_y_mm, actual_steps, target_steps, dt);

    // Command the steppers directly in STEP space (no angle conversion needed).
    pos[0] = (long)target_steps[0];
    pos[1] = (long)target_steps[1];
    pos[2] = (long)target_steps[2];
    motorA.moveTo(pos[0]);
    motorB.moveTo(pos[1]);
    motorC.moveTo(pos[2]);

    // Reuse the proportional-speed smoother from the PID build.
    speed_controller();

}