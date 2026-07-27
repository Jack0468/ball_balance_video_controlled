// ---------------------------------------------------------------------------
// TouchProbe.cpp -- touchscreen ground truth, streamed to the PC.
// See TouchProbe.h for the protocol and the design constraints.
//
// The calibration block below is lifted verbatim from the old Screen.cpp so the
// numbers you already trusted stay trusted. The one thing you MUST check is the
// FRAME MATCH note further down.
// ---------------------------------------------------------------------------

#include "TouchProbe.h"
#include "Screen.h"
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

// ---------------------------------------------------------------------------
// >>> FRAME MATCH -- READ THIS BEFORE YOU TRUST A SINGLE CSV ROW <<<
//
// The comparison is only meaningful if touchscreen mm and vision mm describe
// the same physical frame. Right now they DO NOT, by construction:
//
//   vision:       main.py maps the plate corners to  +/-70.0 x +/-55.0 mm
//                 (HomographyProjector dst_pts)
//   touchscreen:  this file maps the glass to        +/-93.75 x +/-70.5 mm
//
// Those are different rectangles. Either the YOLO corner keypoints sit inboard
// of the touchscreen's active area, or one of the two calibrations is wrong.
// Until you resolve it every error you measure will contain a fixed affine
// term and you will be scoring your calibration, not your model.
//
// Procedure (5 minutes, do it once):
//   1. Flash this, run main.py with --log_csv.
//   2. Rest the ball at 5 known spots: dead centre and near each corner.
//   3. Fit  touch = A * vision + b  over those points (tools/fit_frames.py in
//      the CSV notes, or just least-squares in numpy).
//   4. Put the result in TOUCH_GAIN_* / TOUCH_OFFSET_*_MM below, OR -- better --
//      leave these at identity and apply the fit on the host, so the raw
//      sensor stream stays untouched in the log. Identity is the default.
// ---------------------------------------------------------------------------
#define TOUCH_GAIN_X      1.0
#define TOUCH_GAIN_Y      1.0
#define TOUCH_OFFSET_X_MM 0.0
#define TOUCH_OFFSET_Y_MM 0.0

// ------------------------------ sampling -----------------------------------
// TOUCH_RATE_HZ telemetry records per second. Each record is the median of
// TOUCH_SAMPLES raw reads, and those reads are spread across the frame period
// (one per call) so we never block long enough to jitter a step pulse.
//
// Budget check: worst case AccelStepper speed here is 1300 steps/s = 770 us per
// step. One getPoint() is ~300-500 us, so a single read fits inside one step
// period. Taking all 3 back-to-back would not -- hence the interleave.
#define TOUCH_RATE_HZ   60
#define TOUCH_SAMPLES   3            // must be odd; set to 1 to disable median

#define TOUCH_PERIOD_MS   (1000 / TOUCH_RATE_HZ)
#define TOUCH_SUBPERIOD_MS (TOUCH_PERIOD_MS / TOUCH_SAMPLES)

// Emit a record even when no ball is present. You want the dropouts in the log:
// "vision saw a ball, touchscreen saw nothing" is exactly a false positive.
#define TOUCH_EMIT_WHEN_LOST 1

// Never block on a full USB CDC TX buffer. If the host stopped reading we drop
// the line and carry on. Set to 0 only if your core's availableForWrite() lies.
#define TOUCH_TX_NONBLOCKING 1
#define TOUCH_LINE_MAX 96

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

    last_touch_x     = x_mm * TOUCH_GAIN_X + TOUCH_OFFSET_X_MM;
    last_touch_y     = y_mm * TOUCH_GAIN_Y + TOUCH_OFFSET_Y_MM;
    last_touch_valid = true;
  } else {
    // Hold the last position, flag it invalid. Same convention the vision path
    // uses, so the host sees one consistent semantics on both streams.
    last_touch_valid = false;
  }

  n_raw = 0;

#if !TOUCH_EMIT_WHEN_LOST
  if (!last_touch_valid) return;
#endif

  // Echo what the control loop is currently acting on, so one CSV row is
  // self-contained and the host never has to guess which frame it belongs to.
  coords c = get_coords();

  char line[TOUCH_LINE_MAX];
  int len = snprintf(line, sizeof(line),
                     "T,%lu,%lu,%ld,%ld,%d,%ld,%ld,%ld,%ld,%ld\n",
                     (unsigned long)serial_coords_seq(),
                     (unsigned long)millis(),
                     to_centi(last_touch_x),
                     to_centi(last_touch_y),
                     last_touch_valid ? 1 : 0,
                     to_centi(c.x_mm),
                     to_centi(c.y_mm),
                     motorA.currentPosition(),
                     motorB.currentPosition(),
                     motorC.currentPosition());

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