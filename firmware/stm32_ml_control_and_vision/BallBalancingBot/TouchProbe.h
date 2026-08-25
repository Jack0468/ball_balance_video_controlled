#ifndef TOUCHPROBE_H
#define TOUCHPROBE_H

#include <Arduino.h>

// ---------------------------------------------------------------------------
// TouchProbe -- the UPLINK half of the loop.
//
// The resistive touchscreen under the plate is no longer a control input. It is
// now an independent ground-truth sensor: we sample it, convert to the same
// millimetre frame the vision pipeline uses, and stream it back to the PC so
// the host can log (vision estimate, touchscreen truth) pairs and score the
// models offline.
//
// Design constraints this module respects:
//   * NON-BLOCKING. One ADC pair per call, at most. The stepper ISR-less
//     AccelStepper::run() calls in loop() must not be starved.
//   * NON-BLOCKING TX. Never waits on a full USB CDC buffer; drops the sample
//     instead. A dropped telemetry line must never perturb the control loop.
//   * The control path (SerialCoords -> RLControl) is untouched. If this whole
//     module is deleted the robot still balances exactly as before.
//
// WIRE PROTOCOL, MCU -> PC (ASCII, one sample per line, '\n' terminated):
//
//   T,<seq>,<mcu_ms>,<touch_x>,<touch_y>,<valid>,<vis_x>,<vis_y>,<a>,<b>,<c>
//
//   seq      uint32  vision frame the MCU was acting on when this was sampled
//   mcu_ms   uint32  millis() at sample time
//   touch_x  int     touchscreen X, HUNDREDTHS of a mm   (i.e. mm * 100)
//   touch_y  int     touchscreen Y, HUNDREDTHS of a mm
//   valid    0|1     1 = real contact, 0 = no ball on the plate
//   vis_x    int     vision X the MCU is using, hundredths of a mm  (echo)
//   vis_y    int     vision Y the MCU is using, hundredths of a mm  (echo)
//   a,b,c    long    ACTUAL stepper positions in steps
//
// Everything is sent as INTEGERS on purpose: newlib-nano's printf on STM32
// often ships without float support, so "%f" silently emits garbage. Hundredths
// of a mm is 0.01 mm resolution, far below the touchscreen's noise floor.
//
// Any line the MCU emits that is NOT a "T," record is prefixed with '#' so the
// host parser can skip it without ambiguity.
// ---------------------------------------------------------------------------

void touch_probe_init();

// Call once per loop(). Internally rate-gated; returns immediately most of the
// time. Costs one analog touchscreen read when it does fire (~0.3-0.5 ms).
void touch_probe_update();

// Last median-filtered touchscreen reading, in mm, plate-centre origin.
// Useful if you ever want an on-MCU sanity check or a local safety cutout.
bool  touch_probe_last(double *x_mm, double *y_mm);

#endif