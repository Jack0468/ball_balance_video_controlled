// ---------------------------------------------------------------------------
// SerialCoords.cpp  --  drop-in replacement for Screen.cpp
//
// Instead of reading the resistive touchscreen, ball coordinates arrive over a
// serial port from a host (Python vision script, another MCU, etc.).
// Implements the exact same API declared in Screen.h, so RLControl.cpp and
// BallBalancingBot.ino need no changes.
//
// WIRE PROTOCOL (ASCII, one sample per line, '\n' or '\r\n' terminated):
//
//     "<x_mm>,<y_mm>"            ball detected at x,y   e.g.  "-12.5,43.0"
//     "<x_mm>,<y_mm>,<z>"        z <= 0 means "not detected"
//     "L"  or  "N"               ball lost / out of frame
//
// Separators may be comma, semicolon, space or tab. Coordinates are in
// MILLIMETRES with (0,0) at the centre of the plate -- same convention the
// touchscreen version produced and the RL policy was trained on.
//
// If no valid sample arrives for COORD_TIMEOUT_MS the ball is reported as lost
// (z = 0) while the last known position is still returned, mirroring the old
// Screen.cpp behaviour. rl_balance() then levels the plate after 3 s.
// ---------------------------------------------------------------------------

#include "Screen.h"
#include <stdlib.h>

// --------------------------- configuration ---------------------------------

// Which port the host talks on. Serial is already opened at 2 Mbaud in the
// sketch. Change to Serial1/Serial2 if you want coordinates on a hardware UART
// separate from the debug console.
#define COORD_SERIAL       Serial

// Set to 1 and pick a baud if COORD_SERIAL is NOT the one the .ino opens.
#define COORD_BEGIN_SERIAL 0
#define COORD_BAUD         2000000

// Sample is considered stale after this long without a valid line.
// At 30 Hz control cadence, 150 ms = ~4 missed frames.
#define COORD_TIMEOUT_MS   150

#define COORD_LINE_MAX     48      // longest accepted line, incl. terminator

// Sign / scale fixes, in case the host's axes disagree with the plate.
#define COORD_INVERT_X     0
#define COORD_INVERT_Y     0
#define COORD_SCALE_X      1.0
#define COORD_SCALE_Y      1.0

// Optional: clamp to the physical plate so a garbage frame can't fling the
// plate. Set to 0 to disable.
#define COORD_CLAMP        1
#define COORD_LIMIT_X_MM   93.75   // half of 187.5
#define COORD_LIMIT_Y_MM   70.5    // half of 141.0

// ----------------------------- state ---------------------------------------

static char          line_buf[COORD_LINE_MAX];
static uint8_t       line_len   = 0;
static bool          discarding = false;   // true while flushing an over-long line

static double        last_x     = 0.0;
static double        last_y     = 0.0;
static double        last_tx    = 0.0;
static double        last_ty    = 0.0;
static bool          has_ball   = false;
static bool          has_new    = false;   // a fresh line arrived since last check
static unsigned long last_good_ms = 0;

// ------------------------- helpers (unchanged API) -------------------------

// map command but can return floating point values
double mapf(double x, double in_min, double in_max, double out_min, double out_max) {
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

static inline bool is_sep(char c) {
  return (c == ',' || c == ';' || c == ' ' || c == '\t');
}

// Parse one complete, NUL-terminated line into the state above.
static void handle_line(char *line) {
  char *p = line;
  char *end;

  while (is_sep(*p)) p++;
  if (*p == '\0' || *p == '#') return;                 // blank or comment

  // Explicit "lost" markers.
  if (*p == 'L' || *p == 'l' || *p == 'N' || *p == 'n') {
    has_ball = false;
    has_new  = true;
    return;
  }

  double x = strtod(p, &end);
  if (end == p) return;                                // not a number -> ignore
  p = end;

  while (is_sep(*p)) p++;
  double y = strtod(p, &end);
  if (end == p) return;                                // incomplete pair
  p = end;

  // Parse Target X
  while (is_sep(*p)) p++;
  end = p;
  double tx = strtod(p, &end);
  if (end == p) {
      // If only 2 floats were sent, assume target is (0,0) and there is no Z
      tx = 0.0;
  } else {
      p = end;
  }

  // Parse Target Y
  while (is_sep(*p)) p++;
  end = p;
  double ty = strtod(p, &end);
  if (end == p) {
      ty = 0.0;
  } else {
      p = end;
  }

  // Optional fifth field: <= 0 means the host saw nothing.
  while (is_sep(*p)) p++;
  end = p;
  double z = strtod(p, &end);
  if (end != p && z <= 0.0) {
    has_ball = false;
    has_new  = true;
    return;
  }

  x *= COORD_SCALE_X;
  y *= COORD_SCALE_Y;
#if COORD_INVERT_X
  x = -x;
#endif
#if COORD_INVERT_Y
  y = -y;
#endif
#if COORD_CLAMP
  if (x >  COORD_LIMIT_X_MM) x =  COORD_LIMIT_X_MM;
  if (x < -COORD_LIMIT_X_MM) x = -COORD_LIMIT_X_MM;
  if (y >  COORD_LIMIT_Y_MM) y =  COORD_LIMIT_Y_MM;
  if (y < -COORD_LIMIT_Y_MM) y = -COORD_LIMIT_Y_MM;
#endif

  last_x       = x;
  last_y       = y;
  last_tx      = tx;
  last_ty      = ty;
  has_ball     = true;
  has_new      = true;
  last_good_ms = millis();
}

// Non-blocking: drain whatever bytes are waiting. Safe to call as often as you
// like; get_coords()/check_detected() call it themselves, but calling it from
// loop() too keeps the RX buffer from overflowing at high frame rates.
void serial_coords_poll() {
  while (COORD_SERIAL.available()) {
    char c = (char)COORD_SERIAL.read();

    if (c == '\n' || c == '\r') {
      if (!discarding && line_len > 0) {
        line_buf[line_len] = '\0';
        handle_line(line_buf);
      }
      line_len   = 0;
      discarding = false;
    } else if (discarding) {
      // keep dropping until the line ends
    } else if (line_len < COORD_LINE_MAX - 1) {
      line_buf[line_len++] = c;
    } else {
      line_len   = 0;
      discarding = true;                               // over-long line, drop it
    }
  }

  // Age out a stale sample.
  if (has_ball && (millis() - last_good_ms) > COORD_TIMEOUT_MS) {
    has_ball = false;
  }
}

// True once per received sample: this is what paces the control loop now.
// Polls internally, so it is safe to call as the only serial entry point.
// A burst of backlogged lines collapses into a single "true" -- only the most
// recent coordinate survives, which is what you want after a stall.
bool coords_available() {
  serial_coords_poll();
  bool n = has_new;
  has_new = false;
  return n;
}

// ------------------------------ public API ---------------------------------

// Kept for source compatibility with Screen.cpp; no touchscreen pins to set up.
void screen_init() {
#if COORD_BEGIN_SERIAL
  COORD_SERIAL.begin(COORD_BAUD);
#endif
  line_len     = 0;
  discarding   = false;
  has_ball     = false;
  has_new      = false;
  last_x       = 0.0;
  last_y       = 0.0;
  last_tx      = 0.0;
  last_ty      = 0.0;
  last_good_ms = millis();
}

// checks whether the host currently reports a ball
bool check_detected() {
  serial_coords_poll();
  return has_ball;
}

// returns coordinates of the ball's position
coords get_coords() {
  serial_coords_poll();

  coords p;
  p.x_mm = last_x;                 // last known good position either way
  p.y_mm = last_y;
  p.target_x_mm = last_tx;
  p.target_y_mm = last_ty;
  p.z    = has_ball ? 1.0 : 0.0;   // 0 signals NO BALL to the controller
  return p;
}