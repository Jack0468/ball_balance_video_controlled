#include "RemoteStepControl.h"
#include "MotorControl.h"
#include <Arduino.h>
#include <stdlib.h>

// --------------------------- configuration ---------------------------------

#define REMOTE_SERIAL       Serial
#define REMOTE_LINE_MAX      48   // longest accepted line, incl. terminator

// Safety clamp on incoming step targets. Set equal to the control net's own
// action bound (MAX_MOTOR_STEP in control_net.py / RLControl.cpp) as a sane
// conservative default -- review against the platform's real mechanical
// travel limits before trusting this for anything but the same control net
// that's already known to respect this bound.
#define REMOTE_STEP_LIMIT    98

// A sample is "fresh" for this long; beyond it we hold the last target rather
// than drift. Matches SerialCoords.cpp's COORD_TIMEOUT_MS convention (a
// couple of missed ~30Hz cycles).
#define REMOTE_HOLD_TIMEOUT_MS   200

// Beyond this long with no fresh sample, level the plate -- mirrors
// RLControl.cpp's "ball lost for 3s -> level" safety behaviour exactly, so
// this mode fails safe the same way the RL mode does.
#define REMOTE_LEVEL_TIMEOUT_MS  3000

// ----------------------------- state ---------------------------------------
// pos[3], motorA/B/C, and speed_controller() all come from MotorControl.h,
// already included above -- no need to re-declare them here.

static char          line_buf[REMOTE_LINE_MAX];
static uint8_t        line_len   = 0;
static bool           discarding = false;

static long           last_steps[3] = {0, 0, 0};
static bool           has_new       = false;
static unsigned long  last_good_ms  = 0;
static bool           leveled       = false;  // avoid re-issuing moveTo(0) every loop while stale

static inline bool is_sep(char c) {
  return (c == ',' || c == ';' || c == ' ' || c == '\t');
}

// Parse one complete, NUL-terminated "S,<a>,<b>,<c>" line into last_steps[].
// Malformed or out-of-range lines are silently dropped (not applied) --
// mirrors SerialCoords.cpp's "not a number -> ignore" behaviour rather than
// crashing or applying a partially-parsed target.
static void handle_line(char *line) {
  char *p = line;
  while (is_sep(*p)) p++;

  if (*p != 'S' && *p != 's') return;  // not our format -- ignore
  p++;
  while (is_sep(*p)) p++;

  long parsed[3];
  for (int i = 0; i < 3; i++) {
    char *end;
    long v = strtol(p, &end, 10);
    if (end == p) return;  // incomplete triple -- drop the whole line
    parsed[i] = v;
    p = end;
    while (is_sep(*p)) p++;
  }

  for (int i = 0; i < 3; i++) {
    if (parsed[i] > REMOTE_STEP_LIMIT) parsed[i] = REMOTE_STEP_LIMIT;
    if (parsed[i] < -REMOTE_STEP_LIMIT) parsed[i] = -REMOTE_STEP_LIMIT;
    last_steps[i] = parsed[i];
  }

  has_new      = true;
  last_good_ms = millis();
  leveled      = false;
}

static void remote_step_poll() {
  while (REMOTE_SERIAL.available()) {
    char c = (char)REMOTE_SERIAL.read();

    if (c == '\n' || c == '\r') {
      if (!discarding && line_len > 0) {
        line_buf[line_len] = '\0';
        handle_line(line_buf);
      }
      line_len   = 0;
      discarding = false;
    } else if (discarding) {
      // keep dropping until the line ends
    } else if (line_len < REMOTE_LINE_MAX - 1) {
      line_buf[line_len++] = c;
    } else {
      line_len   = 0;
      discarding = true;  // over-long line, drop it
    }
  }
}

// ------------------------------ public API ---------------------------------

void remote_step_control_init() {
  line_len     = 0;
  discarding   = false;
  has_new      = false;
  leveled      = false;
  last_steps[0] = last_steps[1] = last_steps[2] = 0;
  last_good_ms = millis();
}

bool remote_step_target_available() {
  remote_step_poll();
  bool n = has_new;
  has_new = false;
  return n;
}

void remote_step_control_update() {
  bool fresh = remote_step_target_available();
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

  if (age >= REMOTE_LEVEL_TIMEOUT_MS) {
    // Stale for too long -- fail safe by levelling, same policy as
    // RLControl.cpp's ball-lost handling. Only issue the command once per
    // stale period, not every loop iteration.
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

  // Within REMOTE_HOLD_TIMEOUT_MS..REMOTE_LEVEL_TIMEOUT_MS: hold the last
  // commanded target (do nothing) rather than drift or re-command.
}
