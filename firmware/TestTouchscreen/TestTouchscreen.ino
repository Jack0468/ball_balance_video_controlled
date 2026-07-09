#include <TouchScreen.h>

// Touchscreen wiring for STM32F407
#define YP PA2  // Must be an analog pin (ADC123_IN2)
#define XM PA3  // Must be an analog pin (ADC123_IN3)
#define YM PB0  // Can be digital (but is also ADC capable)
#define XP PB1  // Can be digital (but is also ADC capable)

// The third parameter is the resistance across the X plate (usually around 300 ohms)
TouchScreen ts = TouchScreen(XP, YP, XM, YM, 300);

void setup() {
  Serial.begin(115200);
  
  // Give some time for the serial monitor to open
  delay(2000); 
  Serial.println("--- STM32 Resistive Touchscreen Test ---");
  Serial.println("Waiting for touches...");
}

void loop() {
  // Retrieve the point from the touchscreen
  TSPoint p = ts.getPoint();
  
  // We have some minimum pressure we consider 'valid' (z > 10)
  // If z is 0, no one is pressing the screen
  if (p.z > 10 && p.z < 1000) {
     Serial.print("Raw X = "); Serial.print(p.x);
     Serial.print("\tRaw Y = "); Serial.print(p.y);
     Serial.print("\tPressure (Z) = "); Serial.println(p.z);
  }

  delay(50); // Small delay to avoid flooding the serial monitor
}
