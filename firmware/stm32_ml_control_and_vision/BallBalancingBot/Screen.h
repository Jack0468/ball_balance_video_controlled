#ifndef SCREEN_H
#define SCREEN_H

#include <Arduino.h>
#include <TouchScreen.h>

// ---------------------------------------------------------------------------
// "Screen" is now the DOWNLINK: ball + target coordinates that the PC's vision
// pipeline pushes to the MCU over serial (see SerialCoords.cpp). It is what the
// RL policy consumes.
//
// The physical resistive touchscreen is still wired up, but it is no longer the
// control input -- it is the GROUND-TRUTH sensor, sampled and streamed back up
// to the PC by TouchProbe.cpp. See TouchProbe.h.
// ---------------------------------------------------------------------------

struct coords {
  double x_mm;
  double y_mm;
  double target_x_mm;
  double target_y_mm;
  double z;            // >0 = ball present, 0 = lost / stale
};

double mapf(double x, double in_min, double in_max, double out_min, double out_max);
void screen_init();
bool check_detected();
coords get_coords();

// Serial coords (downlink) plumbing
void serial_coords_poll();
bool coords_available();

// NEW: sequence number of the most recent vision frame accepted from the PC.
// TouchProbe echoes this back so the host can join its own vision estimate to
// the touchscreen reading taken while that estimate was in force.
uint32_t serial_coords_seq();

#endif