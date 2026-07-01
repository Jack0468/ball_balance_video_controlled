#include <Arduino.h>
#include "CameraAcquisition.h"
#include "Screen.h"
#include "MotorControl.h"
#include "InverseKinematics.h"
#include "PIDControllers.h"

// Globals needed by PIDControllers
bool enable_serial_output = false; // Disable text output so we can stream binary frames

// State variables
double target_x = 0;
double target_y = 0;

void parseDummyTargets() {
  // Simple non-blocking serial parser to read dummy targets from Python
  // Expected format: T<target_x>,<target_y>\n
  // Example: T0.0,0.0\n
  while (Serial.available() > 0) {
    char c = Serial.read();
    static String buffer = "";
    if (c == '\n') {
      if (buffer.startsWith("T") && buffer.indexOf(',') != -1) {
        int tComma = buffer.indexOf(',');
        if (tComma != -1) {
          target_x = buffer.substring(1, tComma).toDouble();
          target_y = buffer.substring(tComma + 1).toDouble();
        }
      }
      buffer = "";
    } else {
      buffer += c;
    }
  }
}

void setup() {
  Serial.begin(2000000); // 2 Mbps for high-speed streaming
  enable_serial_output = false;
  
  // Initialize sensors
  camera_init();
  screen_init();
  
  // Initialize motors
  motor_init();
  home_motors();
  go_home();
  
  delay(1000);
}

void loop() {
  // 1. Read incoming dummy targets from Python
  parseDummyTargets();

  // 2. Capture a frame from the OV7670 camera
  if (captureFrame()) {
    
    // 3. Poll the resistive touchscreen immediately after frame acquisition
    coords p = get_coords();
    float touch_x = (float)p.x_mm;
    float touch_y = (float)p.y_mm;

    // 4. Run the PID control loop with the dummy targets and local touch coordinates
    pid_balance_with_coords(target_x, target_y, p.x_mm, p.y_mm);

    // 5. Serialize and send the dual-sensor payload
    const uint8_t syncHeader[] = {0xAA, 0xBB, 0xCC, 0xDD};
    
    Serial.write(syncHeader, 4);
    Serial.write((uint8_t*)&touch_x, sizeof(float));
    Serial.write((uint8_t*)&touch_y, sizeof(float));
    Serial.write((uint8_t*)frameBuffer, sizeof(frameBuffer));
    
    Serial.send_now();
  } else {
    delay(500); 
  }
}
