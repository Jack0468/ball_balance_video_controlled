#include <Arduino.h>
#include "InverseKinematics.h"
#include "MotorControl.h"
#include "PIDControllers.h"
#include "CameraAcquisition.h"

// Global variable declaration
bool enable_serial_output = false; // Disable text output so we can stream binary frames

// State variables received from Python Multiplexer
double target_x = 0;
double target_y = 0;
double current_x = 0;
double current_y = 0;

// Function definitions
void muteAllSerialOutput() {
  enable_serial_output = false;
}

void enableAllSerialOutput() {
  enable_serial_output = true;
}

void parseSerialData() {
  // Simple non-blocking serial parser to read coordinates from Multiplexer
  // Expected format: T<target_x>,<target_y>C<current_x>,<current_y>\n
  // Example: T0.0,0.0C12.5,-4.2\n
  while (Serial.available() > 0) {
    char c = Serial.read();
    static String buffer = "";
    if (c == '\n') {
      if (buffer.startsWith("T") && buffer.indexOf('C') != -1) {
        int cIndex = buffer.indexOf('C');
        String tPart = buffer.substring(1, cIndex);
        String cPart = buffer.substring(cIndex + 1);
        
        int tComma = tPart.indexOf(',');
        int cComma = cPart.indexOf(',');
        
        if (tComma != -1 && cComma != -1) {
          target_x = tPart.substring(0, tComma).toDouble();
          target_y = tPart.substring(tComma + 1).toDouble();
          current_x = cPart.substring(0, cComma).toDouble();
          current_y = cPart.substring(cComma + 1).toDouble();
        }
      }
      buffer = "";
    } else {
      buffer += c;
    }
  }
}

void setup() {
  Serial.begin(2000000); // Increased baud rate for faster binary transfer
  muteAllSerialOutput();
  
  // screen_init(); // Removed in favor of Camera/ML Vision
  camera_init();
  
  motor_init();
  home_motors();
  go_home();
  delay(1000);
}

void loop() {
  // 1. Capture frame and stream to host
  if (captureFrame()) {
    const uint8_t syncHeader[] = {0xAA, 0xBB, 0xCC, 0xDD};
    Serial.write(syncHeader, 4);
    Serial.write((uint8_t*)frameBuffer, sizeof(frameBuffer));
    Serial.send_now();
  }
  
  // 2. Read incoming commands/coordinates from Python Multiplexer
  parseSerialData();
  
  // 3. Run PID balance loop with updated coordinates
  // Note: pid_balance needs to be updated to use these global coordinates
  // instead of calling get_coords() internally.
  pid_balance_with_coords(target_x, target_y, current_x, current_y);
}