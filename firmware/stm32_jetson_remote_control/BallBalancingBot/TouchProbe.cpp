// ---------------------------------------------------------------------------
// TouchProbe.cpp -- touchscreen ground truth, streamed to the Jetson.
// Jetson-remote-control variant -- see TouchProbe.h for what differs from
// the stm32_ml_control_and_vision/ original and why (no SerialCoords
// dependency, no vis_x/vis_y echo).
//
// The calibration block below is lifted verbatim from the original so the
// numbers you already trusted stay trusted.
// ---------------------------------------------------------------------------

#include "TouchProbe.h"
#include "MotorControl.h"
#include <TouchScreen.h>
#include <stdio.h>
#include <stdlib.h>   // qsort
#include <string.h>   // memcpy

// ------------------------- touchscreen wiring ------------------------------
// (lettering on the ribbon pin is the underside, red wire goes to 14)
// None of these collide with the stepper pins (PD0..PD7) in MotorControl.cpp.
#define YP PA2  // Must be an analog pin (ADC123_IN2)
#define XM PA3  // Must be an analog pin (ADC123_IN3)
#define YM PB0  // Can be digital (but is also ADC capable)
#define XP PB1  // Can be digital (but is also ADC capable)

// ------------------------- touchscreen calibration -------------------------
// The touchscreen axes are physically rotated 90 degrees relative to the plate:
// raw Y drives physical left/right, raw X drives physical top/bottom.
#define TS_LEFT   957   // Raw Y at Left Edge
#define TS_RIGHT   54   // Raw Y at Right Edge
#define TS_TOP    936   // Raw X at Top Edge
#define TS_BOTTOM  92   // Raw X at Bottom Edge

#define SCREEN_WIDTH_MM  187.5
#define SCREEN_HEIGHT_MM 141.0

// Contact threshold. Below this the STM32 ADC is just reading float noise.
#define TS_PRESSURE_MIN 3

// mapf() lived in Screen.h/SerialCoords.cpp in the original firmware; that
// module isn't compiled into this one (see TouchProbe.h), so it's inlined
// here -- same one-line implementation, no behaviour change.
static double mapf(double x, double in_min, double in_max, double out_min, double out_max) {
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

// ------------------------------ sampling -----------------------------------
// Same rate/rationale as the original -- see docs/PROJECT_LOGBOOK.md's 19/08
// entry on why 25Hz, not 60Hz.
#define TOUCH_RATE_HZ   25
#define TOUCH_SAMPLES   3            // must be odd; set to 1 to disable median

#define TOUCH_PERIOD_MS   (1000 / TOUCH_RATE_HZ)
#define TOUCH_SUBPERIOD_MS (TOUCH_PERIOD_MS / TOUCH_SAMPLES)

// Emit a record even when no ball is present. Dropouts are meaningful data.
#define TOUCH_EMIT_WHEN_LOST 1

// Never block on a full USB CDC TX buffer. If the host stopped reading we drop
// the line and carry on.
#define TOUCH_TX_NONBLOCKING 1
#define TOUCH_LINE_MAX 80

#define TOUCH_SERIAL Serial

// ------------------------------- state -------------------------------------

static TouchScreen ts = TouchScreen(XP, YP, XM, YM, 300);  // 300 = plate ohms

static int16_t  raw_x[TOUCH_SAMPLES];
static int16_t  raw_y[TOUCH_SAMPLES];
static uint8_t  raw_valid[TOUCH_SAMPLES];
static uint8_t  n_raw = 0;

static unsigned long next_sample_ms = 0;

static double   last_touch_x = 0.0;
static double   last_touch_y = 0.0;
static bool     last_touch_valid = false;

static uint32_t dropped_tx = 0;      // telemetry lines lost to a full TX buffer
static uint32_t local_seq  = 0;      // local record counter -- see TouchProbe.h

// --------------------------- helpers ---------------------------------------

static int cmp_i16(const void *a, const void *b) {
  return (*(const int16_t *)a) - (*(const int16_t *)b);
}

static int16_t median_i16(int16_t *v, uint8_t n) {
  if (n == 1) return v[0];
  int16_t tmp[TOUCH_SAMPLES];
  memcpy(tmp, v, n * sizeof(int16_t));
  qsort(tmp, n, sizeof(int16_t), cmp_i16);
  return tmp[n / 2];
}

// Round-half-away-from-zero to hundredths, as an integer.
static long to_centi(double mm) {
  return (long)(mm * 100.0 + (mm >= 0 ? 0.5 : -0.5));
}

// One raw touchscreen read. This is the only place that blocks, and only for
// the duration of two analogReads.
static void take_raw_sample() {
  TSPoint p = ts.getPoint();

  // getPoint() leaves the shared pins configured for reading; restore them.
  pinMode(XM, OUTPUT);
  pinMode(YP, OUTPUT);

  raw_x[n_raw]     = p.x;
  raw_y[n_raw]     = p.y;
  raw_valid[n_raw] = (p.z > TS_PRESSURE_MIN) ? 1 : 0;
  n_raw++;
}

static void emit_record() {
  // Majority vote on contact: 2-of-3 rejects a single dropout without adding
  // latency the way a time-based debounce would.
  uint8_t votes = 0;
  for (uint8_t i = 0; i < n_raw; i++) votes += raw_valid[i];
  bool valid = (votes * 2 > n_raw);

  if (valid) {
    int16_t mx = median_i16(raw_x, n_raw);
    int16_t my = median_i16(raw_y, n_raw);

    // Raw Y -> physical X, raw X -> physical Y (the 90 degree rotation), and
    // the trailing sign flip that the old Screen.cpp applied.
    double x_mm = -mapf(my, TS_LEFT,   TS_RIGHT, -SCREEN_WIDTH_MM  / 2.0, SCREEN_WIDTH_MM  / 2.0);
    double y_mm = -mapf(mx, TS_BOTTOM, TS_TOP,   -SCREEN_HEIGHT_MM / 2.0, SCREEN_HEIGHT_MM / 2.0);

    last_touch_x     = x_mm;
    last_touch_y     = y_mm;
    last_touch_valid = true;
  } else {
    // Hold the last position, flag it invalid.
    last_touch_valid = false;
  }

  n_raw = 0;

#if !TOUCH_EMIT_WHEN_LOST
  if (!last_touch_valid) return;
#endif

  char line[TOUCH_LINE_MAX];
  int len = snprintf(line, sizeof(line),
                     "T,%lu,%lu,%ld,%ld,%d,%ld,%ld,%ld\n",
                     (unsigned long)local_seq,
                     (unsigned long)millis(),
                     to_centi(last_touch_x),
                     to_centi(last_touch_y),
                     last_touch_valid ? 1 : 0,
                     motorA.currentPosition(),
                     motorB.currentPosition(),
                     motorC.currentPosition());
  local_seq++;

  if (len <= 0 || len >= (int)sizeof(line)) return;   // truncated -> drop

#if TOUCH_TX_NONBLOCKING
  if (TOUCH_SERIAL.availableForWrite() < len) {
    dropped_tx++;
    return;                                            // host is not draining
  }
#endif
  TOUCH_SERIAL.write((const uint8_t *)line, len);
}

// ------------------------------ public API ---------------------------------

void touch_probe_init() {
  pinMode(XM, OUTPUT);
  pinMode(YP, OUTPUT);
  n_raw            = 0;
  next_sample_ms   = millis();
  last_touch_x     = 0.0;
  last_touch_y     = 0.0;
  last_touch_valid = false;
  dropped_tx       = 0;
  local_seq        = 0;
}

void touch_probe_update() {
  unsigned long now = millis();
  if ((long)(now - next_sample_ms) < 0) return;        // not yet; cheap exit

  take_raw_sample();

  if (n_raw >= TOUCH_SAMPLES) {
    emit_record();
    // Re-anchor to the frame grid rather than to 'now', so telemetry timestamps
    // stay evenly spaced even if a loop iteration ran long.
    next_sample_ms += TOUCH_PERIOD_MS - (TOUCH_SAMPLES - 1) * TOUCH_SUBPERIOD_MS;
  } else {
    next_sample_ms += TOUCH_SUBPERIOD_MS;
  }

  // If we fell far behind (long blocking call somewhere), resync instead of
  // spinning to catch up.
  if ((long)(millis() - next_sample_ms) > (long)(4 * TOUCH_PERIOD_MS)) {
    next_sample_ms = millis();
    n_raw = 0;
  }
}

bool touch_probe_last(double *x_mm, double *y_mm) {
  if (x_mm) *x_mm = last_touch_x;
  if (y_mm) *y_mm = last_touch_y;
  return last_touch_valid;
}
