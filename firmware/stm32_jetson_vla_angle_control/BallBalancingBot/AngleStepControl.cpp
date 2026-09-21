#include "AngleStepControl.h"
#include "MotorControl.h"
#include <Arduino.h>
#include <stdlib.h>

// --------------------------- configuration ---------------------------------

#define ANGLE_SERIAL       Serial
#define ANGLE_LINE_MAX      64   // longest accepted line, incl. terminator
                                  // (longer than RemoteStepControl's 48 --
                                  // signed decimal degrees, e.g. "-11.025",
                                  // are wider than signed integer steps)

// Safety bound on incoming per-motor angles, in degrees relative to each
// motor's homed zero. See AngleStepControl.h for the full derivation --
// short version: this is MAX_MOTOR_STEP = 98 steps (RLControl.cpp /
// control_net.py / RemoteStepControl.cpp's REMOTE_STEP_LIMIT, the one real
// validated bound in this codebase) converted through MotorControl.cpp's own
// angle_to_steps() convention (3200 steps/rev, 360 deg/rev): 98 * (360.0 /
// 3200.0) = 11.025 degrees. NOT an independently re-derived mechanical
// hardstop spec -- tighten if one is ever documented, do not widen blindly.
#define MAX_MOTOR_ANGLE_DEG  11.025

// Defense-in-depth: after angle_to_steps() conversion, clamp to the same
// step bound the angle limit above was derived from. With an exact input at
// the limit this is a no-op (11.025 deg converts to exactly 98 steps); it
// only guards against float rounding at the boundary, not a substitute for
// the reject-out-of-range policy in handle_line() below.
#define ANGLE_STEP_CLAMP     98L

// A sample is "fresh" for this long; beyond it we hold the last target
// rather than drift (steppers naturally hold torque at their last commanded
// position -- see BallBalancingBot.ino / MotorControl.cpp's ENA handling).
#define ANGLE_HOLD_TIMEOUT_MS   200

// Beyond this long with no fresh sample, level the plate -- fail-safe for a
// genuinely stalled/crashed host, mirroring RLControl.cpp's and
// RemoteStepControl.cpp's "ball/host lost -> level" policy. Deliberately
// MUCH longer than RemoteStepControl.cpp's 3000 ms: that value was tuned for
// the RL control net's ~30 Hz cadence, where 3 s of silence is dozens of
// missed frames and a real sign of a dead host. This firmware's host is a
// large VLM whose per-inference latency is itself multi-second BY DESIGN
// (host_software/ml_jetson_vla/docs/ARM2_MINIMAL_BASELINE_SCOPE.md section 4
// -- "nothing like Track 1's roughly 30 Hz control loop" is the expected,
// reportable finding, not a bug to route around here). A 3 s level-timeout
// would flatten the plate mid-task on every single normal inference cycle,
// which is exactly the "go idle/drop the ball between updates" failure mode
// this firmware was written to avoid. 15 s is a judgment call: generous
// enough to survive one slow inference cycle, short enough to still fail
// safe if the host genuinely disconnects. Retune once real measured
// per-inference latency on this Jetson is available.
#define ANGLE_LEVEL_TIMEOUT_MS  15000

// ----------------------------- state ---------------------------------------
// pos[3], motorA/B/C, and speed_controller() all come from MotorControl.h,
// already included above -- no need to re-declare them here.

static char          line_buf[ANGLE_LINE_MAX];
static uint8_t        line_len   = 0;
static bool           discarding = false;

static long           last_steps[3] = {0, 0, 0};
static bool           has_new       = false;
static unsigned long  last_good_ms  = 0;
static bool           leveled       = false;  // avoid re-issuing moveTo(0) every loop while stale

static inline bool is_sep(char c) {
  return (c == ',' || c == ';' || c == ' ' || c == '\t');
}

// Parse one complete, NUL-terminated "A,<theta_a>,<theta_b>,<theta_c>" line.
// On success, converts to steps and stores into last_steps[]. Malformed OR
// out-of-range lines are entirely dropped (nothing applied, nothing partially
// updated) -- see AngleStepControl.h's "OUT-OF-RANGE POLICY" for why this
// rejects rather than silently clamps, unlike RemoteStepControl.cpp.
static void handle_line(char *line) {
  char *p = line;
  while (is_sep(*p)) p++;

  if (*p != 'A' && *p != 'a') return;  // not our format -- ignore
  p++;
  while (is_sep(*p)) p++;

  double parsed_deg[3];
  for (int i = 0; i < 3; i++) {
    char *end;
    double v = strtod(p, &end);
    if (end == p) return;  // incomplete triple -- drop the whole line
    parsed_deg[i] = v;
    p = end;
    while (is_sep(*p)) p++;
  }

  for (int i = 0; i < 3; i++) {
    if (parsed_deg[i] > MAX_MOTOR_ANGLE_DEG || parsed_deg[i] < -MAX_MOTOR_ANGLE_DEG) {
      // Plain string literals, not the F() macro -- neither of this
      // firmware's proven-to-compile source directories
      // (stm32_jetson_remote_control/, stm32_ml_control_and_vision/) use
      // F(), so it is not assumed supported on this board's Arduino core.
      ANGLE_SERIAL.print("# ERR angle out of range (limit +/-");
      ANGLE_SERIAL.print(MAX_MOTOR_ANGLE_DEG);
      ANGLE_SERIAL.print(" deg): A=");
      ANGLE_SERIAL.print(parsed_deg[0]);
      ANGLE_SERIAL.print(" B=");
      ANGLE_SERIAL.print(parsed_deg[1]);
      ANGLE_SERIAL.print(" C=");
      ANGLE_SERIAL.println(parsed_deg[2]);
      return;  // reject the whole line, apply nothing
    }
  }

  for (int i = 0; i < 3; i++) {
    long steps = angle_to_steps(parsed_deg[i]);
    // Defense-in-depth clamp -- see ANGLE_STEP_CLAMP comment above.
    if (steps > ANGLE_STEP_CLAMP) steps = ANGLE_STEP_CLAMP;
    if (steps < -ANGLE_STEP_CLAMP) steps = -ANGLE_STEP_CLAMP;
    last_steps[i] = steps;
  }

  has_new      = true;
  last_good_ms = millis();
  leveled      = false;
}

static void angle_step_poll() {
  while (ANGLE_SERIAL.available()) {
    char c = (char)ANGLE_SERIAL.read();

    if (c == '\n' || c == '\r') {
      if (!discarding && line_len > 0) {
        line_buf[line_len] = '\0';
        handle_line(line_buf);
      }
      line_len   = 0;
      discarding = false;
    } else if (discarding) {
      // keep dropping until the line ends
    } else if (line_len < ANGLE_LINE_MAX - 1) {
      line_buf[line_len++] = c;
    } else {
      line_len   = 0;
      discarding = true;  // over-long line, drop it
    }
  }
}

// ------------------------------ public API ---------------------------------

void angle_step_control_init() {
  line_len     = 0;
  discarding   = false;
  has_new      = false;
  leveled      = false;
  last_steps[0] = last_steps[1] = last_steps[2] = 0;
  last_good_ms = millis();
}

bool angle_step_target_available() {
  angle_step_poll();
  bool n = has_new;
  has_new = false;
  return n;
}

void angle_step_control_update() {
  bool fresh = angle_step_target_available();
  unsigned long age = millis() - last_good_ms;

  if (fresh) {
    pos[0] = last_steps[0];
    pos[1] = last_steps[1];
    pos[2] = last_steps[2];
    motorA.moveTo(pos[0]);
    motorB.moveTo(pos[1]);
    motorC.moveTo(pos[2]);
    speed_controller();
    return;
  }

  if (age >= ANGLE_LEVEL_TIMEOUT_MS) {
    // Stale for far too long (well past a single VLM inference cycle) --
    // fail safe by levelling, same policy as RLControl.cpp's ball-lost
    // handling and RemoteStepControl.cpp's stale-host handling. Only issue
    // the command once per stale period, not every loop iteration.
    if (!leveled) {
      pos[0] = pos[1] = pos[2] = 0;
      motorA.moveTo(0);
      motorB.moveTo(0);
      motorC.moveTo(0);
      speed_controller();
      leveled = true;
    }
    return;
  }

  // Within ANGLE_HOLD_TIMEOUT_MS..ANGLE_LEVEL_TIMEOUT_MS (i.e. for the
  // entire span of a normal, even slow, VLM inference cycle): hold the last
  // commanded target -- do nothing here. This is the "keep balancing during
  // inference" requirement: motorA/B/C.run() still runs every loop()
  // iteration (see BallBalancingBot.ino step 3) and the stepper driver stays
  // enabled (MotorControl.cpp's ENA line is set LOW once in motor_init() and
  // never toggled again in this firmware), so the plate holds its last
  // commanded tilt under active torque rather than going slack or
  // re-levelling while the next inference is still running.
}
