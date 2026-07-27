// ---------------------------------------------------------------------------
// SerialCoords.cpp  --  DOWNLINK: ball + target coordinates from the PC.
//
// The PC's vision/audio pipeline pushes the estimated ball position and the
// current target here; the RL policy runs on this. Nothing about that changed.
//
// WHAT CHANGED vs. the previous version
//   1. Lines may now carry a leading "V,<seq>" tag. <seq> is a monotonically
//      increasing frame counter from the host. We store the seq of the most
//      recently accepted frame and expose it via serial_coords_seq(), so the
//      touchscreen telemetry (TouchProbe.cpp) can echo it back and the host can
//      join "what vision thought" to "where the ball actually was".
//   2. The old untagged formats still parse, so an older host keeps working.
//
// WIRE PROTOCOL, PC -> MCU (ASCII, one sample per line, '\n' terminated):
//
//   V,<seq>,<x_mm>,<y_mm>,<tgt_x_mm>,<tgt_y_mm>[,<valid>]   preferred
//   <x_mm>,<y_mm>,<tgt_x_mm>,<tgt_y_mm>[,<valid>]           legacy, still OK
//   <x_mm>,<y_mm>                                            legacy, target=(0,0)
//   L   or   N                                               ball lost
//
// <valid> <= 0 means "the host saw nothing this frame".
// Separators may be comma, semicolon, space or tab. Coordinates are in
// MILLIMETRES with (0,0) at the centre of the plate.
//
// If no valid sample arrives for COORD_TIMEOUT_MS the ball is reported as lost
// (z = 0) while the last known position is still returned. rl_balance() then
// levels the plate after 3 s.
// ---------------------------------------------------------------------------

#include "Screen.h"
#include <stdlib.h>

// --------------------------- configuration ---------------------------------

// Which port the host talks on. Serial is already opened in the sketch.
#define COORD_SERIAL       Serial

// Set to 1 and pick a baud if COORD_SERIAL is NOT the one the .ino opens.
#define COORD_BEGIN_SERIAL 0
#define COORD_BAUD         2000000

// Sample is considered stale after this long without a valid line.
// At 30 Hz control cadence, 150 ms = ~4 missed frames.
#define COORD_TIMEOUT_MS   150

#define COORD_LINE_MAX     64      // longest accepted line, incl. terminator
                                   // (was 48; the "V,<seq>," tag needs room)

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

static uint32_t      last_seq   = 0;       // NEW: host frame counter

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

  // NEW: optional "V,<seq>," header. If absent we just auto-increment, so the
  // host still gets a usable (if less precise) join key.
  bool tagged = false;
  if (*p == 'V' || *p == 'v') {
    p++;
    while (is_sep(*p)) p++;
    unsigned long s = strtoul(p, &end, 10);
    if (end == p) return;                              // "V" with no seq -> junk
    last_seq = (uint32_t)s;
    tagged   = true;
    p = end;
    while (is_sep(*p)) p++;
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

  // Optional trailing validity field: <= 0 means the host saw nothing.
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

  if (!tagged) last_seq++;                             // keep the counter moving

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

// NEW: join key for the host-side accuracy log.
uint32_t serial_coords_seq() {
  return last_seq;
}

// ------------------------------ public API ---------------------------------

// Kept for source compatibility with Screen.cpp; no touchscreen pins to set up
// here -- the touchscreen now belongs to TouchProbe.cpp.
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
  last_seq     = 0;
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