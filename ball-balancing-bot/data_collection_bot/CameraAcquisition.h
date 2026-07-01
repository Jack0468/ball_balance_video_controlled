#ifndef CAMERA_ACQUISITION_H
#define CAMERA_ACQUISITION_H

#include <Arduino.h>
#include <Wire.h>

#define OV7670_I2C_ADDRESS 0x21

// Sync & Clock Pins
#define XCLK_PIN 9   // PTC3
#define VSYNC_PIN 4  // PTA13
#define HREF_PIN 5   // PTD7
#define PCLK_PIN 6   // PTD4

// Image Settings
#define QQVGA_WIDTH 160
#define QQVGA_HEIGHT 120

extern uint16_t frameBuffer[QQVGA_WIDTH * QQVGA_HEIGHT];

void camera_init();
bool captureFrame();

#endif
