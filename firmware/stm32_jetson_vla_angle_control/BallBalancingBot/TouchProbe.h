#ifndef TOUCHPROBE_H
#define TOUCHPROBE_H

#include <Arduino.h>

// ---------------------------------------------------------------------------
// TouchProbe -- the UPLINK half of the loop, Jetson-VLA-angle-control variant
// (firmware/stm32_jetson_vla_angle_control/, copied unchanged from
// firmware/stm32_jetson_remote_control/'s TouchProbe -- see AngleStepControl.h
// for why this directory exists and how its downlink differs).
//
// The resistive touchscreen under the plate is an independent ground-truth
// sensor: sample it, convert to millimetres, and stream it back to the host
// along with the ACTUAL stepper positions -- useful evaluation telemetry for
// any host-side controller driving this firmware (the large-VLM angle
// controller here, or the step-target control_net.py in the sibling
// stm32_jetson_remote_control/ deployment). Telemetry format/behaviour is
// identical to that sibling deployment on purpose -- same evaluation tooling
// (host_software/ml_jetson_vla/runtime/telemetry_logger.py) reads both.
//
// DIFFERS FROM THE stm32_ml_control_and_vision/ VARIANT OF THIS FILE:
// that version echoes the vision (x,y) the SerialCoords downlink most
// recently received (`get_coords()`), so one CSV row is self-contained. In
// this firmware there is no SerialCoords downlink at all -- AngleStepControl.cpp
// is the ONLY thing reading `Serial`, on purpose: a step/angle-target parser
// and SerialCoords's own line reader cannot safely share one Serial RX stream
// in the same loop() (both drain `Serial.available()` independently --
// whichever runs first empties the buffer, silently starving the other, up to
// losing every target command). The Jetson already knows what vision (x,y) it
// computed (or what the VLM's prompt/image were) before sending a target, so
// it doesn't need the MCU to echo it back -- dropping vis_x/vis_y from the
// wire format removes the conflict at the source instead of working around it.
//
// Design constraints this module respects (same as the original):
//   * NON-BLOCKING. One ADC pair per call, at most.
//   * NON-BLOCKING TX. Never waits on a full USB CDC buffer; drops the
//     sample instead. A dropped telemetry line must never perturb control.
//
// WIRE PROTOCOL, MCU -> PC (ASCII, one sample per line, '\n' terminated):
//
//   T,<seq>,<mcu_ms>,<touch_x>,<touch_y>,<valid>,<a>,<b>,<c>
//
//   seq      uint32  local telemetry-record counter (NOT a vision-frame join
//                     key in this firmware -- see note above)
//   mcu_ms   uint32  millis() at sample time
//   touch_x  int     touchscreen X, HUNDREDTHS of a mm   (i.e. mm * 100)
//   touch_y  int     touchscreen Y, HUNDREDTHS of a mm
//   valid    0|1     1 = real contact, 0 = no ball on the plate
//   a,b,c    long    ACTUAL stepper positions in steps (motorX.currentPosition())
//
// Everything is sent as INTEGERS on purpose: newlib-nano's printf on STM32
// often ships without float support. Any line the MCU emits that is NOT a
// "T," record is prefixed with '#' so the host parser can skip it.
// ---------------------------------------------------------------------------

void touch_probe_init();

// Call once per loop(). Internally rate-gated; returns immediately most of the
// time. Costs one analog touchscreen read when it does fire (~0.3-0.5 ms).
void touch_probe_update();

// Last median-filtered touchscreen reading, in mm, plate-centre origin.
bool  touch_probe_last(double *x_mm, double *y_mm);

#endif
