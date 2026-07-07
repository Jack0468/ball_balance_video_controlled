#include <Arduino.h>
#include "InverseKinematics.h"
#include "MotorControl.h"
#include "PIDControllers.h"
#include "Screen.h"
#include "DataCollectionStateMachine.h"

// Global variable declaration
bool enable_serial_output = false; // Disable text output so we can stream binary frames

// State variables
double target_x = 0;
double target_y = 0;
double current_x = 0;
double current_y = 0;

DataCollectionStateMachine state_machine;

#pragma pack(push, 1)
struct TelemetryPacket {
    uint32_t sync_header;
    uint32_t teensy_micros;
    float target_x;
    float target_y;
    float touch_x;
    float touch_y;
    float error_x;
    float error_y;
    float pitch;
    float roll;
    float theta_a;
    float theta_b;
    float theta_c;
    float integral_x;
    float integral_y;
    float deriv_x;
    float deriv_y;
};
#pragma pack(pop)

void muteAllSerialOutput() {
  enable_serial_output = false;
}

void enableAllSerialOutput() {
  enable_serial_output = true;
}

void setup() {
  Serial.begin(2000000); // Fast baud rate for binary transfer
  muteAllSerialOutput();
  
  screen_init(); // Re-enabled for touchscreen ML labels
  
  motor_init();
  home_motors();
  go_home();
  delay(1000);
}

void loop() {
  static unsigned long last_loop_time = 0;
  unsigned long now = millis();
  
  // 50Hz fixed control loop (20ms)
  if (now - last_loop_time >= 20) {
    last_loop_time = now;
    
    // 1. Get current touch coordinates
    coords p = get_coords();
    current_x = p.x_mm;
    current_y = p.y_mm;
    
    // 2. State machine update (Random, Patterns, Sweeps)
    bool is_done = false;
    state_machine.getNextTarget(target_x, target_y, is_done);
    
    // 3. Run PID balance loop with updated coordinates
    pid_balance_with_coords(target_x, target_y, current_x, current_y);
    
    // 4. Send high-frequency telemetry struct over Serial
    TelemetryPacket pkt;
    pkt.sync_header = 0xDDCCBBAA; // 0xAABBCCDD in little endian, typically. We'll unpack it carefully in Python.
    pkt.teensy_micros = micros();
    pkt.target_x = target_x;
    pkt.target_y = target_y;
    pkt.touch_x = current_x;
    pkt.touch_y = current_y;
    
    // In PIDControllers.cpp, index 0 is Y, index 1 is X
    pkt.error_x = error[1];
    pkt.error_y = error[0];
    pkt.pitch = output_angles[1];
    pkt.roll = output_angles[0];
    
    pkt.theta_a = steps_to_angle(pos[0]);
    pkt.theta_b = steps_to_angle(pos[1]);
    pkt.theta_c = steps_to_angle(pos[2]);
    
    pkt.integral_x = integ[1];
    pkt.integral_y = integ[0];
    pkt.deriv_x = deriv[1];
    pkt.deriv_y = deriv[0];
    
    Serial.write((uint8_t*)&pkt, sizeof(TelemetryPacket));
  }
}