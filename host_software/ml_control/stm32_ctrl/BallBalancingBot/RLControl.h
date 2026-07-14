#ifndef RLCONTROL_H
#define RLCONTROL_H

// Replaces the PID controller. Reads the touchscreen, runs the trained
// actor network, and commands the steppers with the network's step output.
//
// Targets are in MILLIMETRES, plate-centre origin (0,0 = middle), matching
// the training environment. The network's input scaling (mm->m, steps/98)
// is folded into weights.h, so this file feeds RAW mm and RAW steps.

// Call once per loop; internally gated to ~30 Hz to match training cadence.
void rl_balance(float target_x_mm, float target_y_mm);

// Low-level: fills target_steps[3] from one observation. Exposed for the
// serial self-test that cross-checks against inference.py.
void rl_infer(float x_mm, float y_mm,
              float target_x_mm, float target_y_mm,
              const float actual_steps[3],
              float target_steps_out[3]);

// Reset the stateful velocity filter (call on ball-lost / re-home).
void rl_reset_state();

#endif // RLCONTROL_H