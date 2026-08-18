// PROPOSAL -- staged for firmware-owner review, see Telemetry.h and
// ../TELEMETRY_PROTOCOL.md. Not yet merged into
// firmware/stm32_ml_control_and_vision/BallBalancingBot/.
//
// Struct/sync-header style deliberately matches the real precedent found in
// firmware/MLVisionPIDControl/PIDControllers.cpp (TelemetryPacket, sent from
// pid_balance()) rather than inventing a new convention -- see Telemetry.h.

#include "Telemetry.h"
#include "MotorControl.h"   // motorA/B/C, steps_to_angle() -- both already exist
#include <Arduino.h>

// --------------------------- configuration ---------------------------------

// Same serial line SerialCoords.cpp/RemoteStepControl.cpp already RX on --
// full-duplex, sending here doesn't disturb inbound parsing on either path.
#define TELEMETRY_SERIAL   Serial

// Packed layout matching TELEMETRY_PROTOCOL.md exactly -- keep these two in
// sync if either changes; the Jetson-side parser (telemetry_logger.py) is
// hand-written against this struct's byte layout, not auto-generated from it.
//
// sync_header as the struct's own first field (not a separately-written
// array) and micros() (not millis()) both match PIDControllers.cpp's
// TelemetryPacket convention exactly -- same 0xDDCCBBAA-in-memory /
// AA-BB-CC-DD-on-the-wire trick (little-endian byte order puts the least
// significant byte, 0xAA, first on the wire).
#pragma pack(push, 1)
struct TelemetryPacket {
  uint32_t sync_header;
  uint32_t mcu_micros;
  float    target_x_mm;
  float    target_y_mm;
  float    touch_x_mm;
  float    touch_y_mm;
  int32_t  actual_step_a;
  int32_t  actual_step_b;
  int32_t  actual_step_c;
  float    theta_a_deg;
  float    theta_b_deg;
  float    theta_c_deg;
};
#pragma pack(pop)

void telemetry_init() {
  // Nothing to configure -- TELEMETRY_SERIAL is already opened by the sketch
  // (BallBalancingBot.ino) at 2,000,000 baud for the existing RX paths.
}

void telemetry_send(float touch_x_mm, float touch_y_mm, float target_x_mm, float target_y_mm) {
  TelemetryPacket pkt;
  pkt.sync_header  = 0xDDCCBBAA;  // -> AA BB CC DD on the wire, see struct comment
  pkt.mcu_micros   = micros();
  pkt.target_x_mm  = target_x_mm;
  pkt.target_y_mm  = target_y_mm;
  pkt.touch_x_mm   = touch_x_mm;
  pkt.touch_y_mm   = touch_y_mm;

  // Real, lagged stepper position -- NOT pos[] (the commanded-target buffer,
  // which is what PIDControllers.cpp's own telemetry uses instead; see the
  // flag in Telemetry.h). Must match what RLControl.cpp itself feeds the
  // control net as actual_steps, or a Jetson-side control net trained on
  // this signal runs out of its training distribution.
  pkt.actual_step_a = (int32_t)motorA.currentPosition();
  pkt.actual_step_b = (int32_t)motorB.currentPosition();
  pkt.actual_step_c = (int32_t)motorC.currentPosition();

  // Derived purely for evaluate_system_control.py's REQUIRED_COLUMNS schema
  // (theta_a/b/c) -- not used by any control path on this device.
  pkt.theta_a_deg = (float)steps_to_angle((int)pkt.actual_step_a);
  pkt.theta_b_deg = (float)steps_to_angle((int)pkt.actual_step_b);
  pkt.theta_c_deg = (float)steps_to_angle((int)pkt.actual_step_c);

  TELEMETRY_SERIAL.write((const uint8_t *)&pkt, sizeof(pkt));
}
