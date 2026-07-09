#include "Screen.h"

// Touchscreen wiring (lettering on the ribbon pin is the underside, red wire goes to 14)
#define YP PA2  // Must be an analog pin (ADC123_IN2)
#define XM PA3  // Must be an analog pin (ADC123_IN3)
#define YM PB0  // Can be digital (but is also ADC capable)
#define XP PB1  // Can be digital (but is also ADC capable)


// Touch screen calibration from physical bounds
// The touchscreen axes are physically rotated 90 degrees relative to the screen dimensions!
#define TS_LEFT 957    // Raw Y at Left Edge
#define TS_RIGHT 54    // Raw Y at Right Edge
#define TS_TOP 936     // Raw X at Top Edge
#define TS_BOTTOM 92   // Raw X at Bottom Edge

// Screen dimensions
#define SCREEN_WIDTH_MM 187.5
#define SCREEN_HEIGHT_MM 141.0

// Pressure thresholds (adjust if needed)
#define MINPRESSURE .000000000001
#define MAXPRESSURE 500

TouchScreen ts = TouchScreen(XP, YP, XM, YM, 300);  // 300 = ohms of touchscreen
TSPoint currentPoint;

//map command but can return floating point values
double mapf(double x, double in_min, double in_max, double out_min, double out_max) {
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

//initializes touchscreen pins
void screen_init() {
  pinMode(XM, OUTPUT);
  pinMode(YP, OUTPUT);
}

//checks if touchscreen detects ball
bool check_detected() {
  currentPoint = ts.getPoint();
  pinMode(XM, OUTPUT);
  pinMode(YP, OUTPUT);

  return (currentPoint.z > 3);
}

//returns coordinates of the ball's position
coords get_coords() {
  static double last_x = 0;
  static double last_y = 0;
  coords p;

  currentPoint = ts.getPoint();
  pinMode(XM, OUTPUT);
  pinMode(YP, OUTPUT);

  // Map raw values to physical mm with (0,0) dead center.
  // Note: Raw Y controls physical Left/Right (X_mm)
  //       Raw X controls physical Top/Bottom (Y_mm)
  double x_mm = mapf(currentPoint.y, TS_LEFT, TS_RIGHT, -SCREEN_WIDTH_MM / 2.0, SCREEN_WIDTH_MM / 2.0);
  double y_mm = mapf(currentPoint.x, TS_BOTTOM, TS_TOP, -SCREEN_HEIGHT_MM / 2.0, SCREEN_HEIGHT_MM / 2.0);

  // The original Teensy code relied on X > 0, but the STM32 ADC floats and reads noise!
  // We must use the Z pressure to filter out phantom touches.
  bool isValidTouch = (currentPoint.z > 3);

  // If no touch is detected, return the last known good position
  if (!isValidTouch) {
    p.x_mm = last_x;
    p.y_mm = last_y;
    p.z = 0; // Signals NO TOUCH to the PID controller
    return p;
  }

  // Valid touch!
  last_x = -x_mm;
  last_y = -y_mm;
  p.x_mm = -x_mm;
  p.y_mm = -y_mm;
  p.z = 1; // Signals TOUCH DETECTED to the PID controller

  return p;
}
