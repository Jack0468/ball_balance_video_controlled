#include <AccelStepper.h>

#define ENA_PIN 8
#define STEP_PIN 2
#define DIR_PIN 3

AccelStepper motorA(1, STEP_PIN, DIR_PIN);

// Based on your MotorControl.cpp, you are using 3200 steps per revolution
// 3200 steps = 360 degrees
// 800 steps = 90 degrees

void setup() {
  Serial.begin(115200);
  Serial.println("--- Stepper Motor Degrees Test ---");

  pinMode(ENA_PIN, OUTPUT);
  digitalWrite(ENA_PIN, LOW);

  motorA.setMinPulseWidth(20);

  // Fast but smooth settings
  motorA.setMaxSpeed(1000);
  motorA.setAcceleration(500);
}

void loop() {
  // Move to 90 degrees
  Serial.println("\nMoving to 90 degrees (800 steps)...");
  motorA.moveTo(800);
  while (motorA.distanceToGo() != 0) motorA.run();
  delay(1000);

  // Move to 180 degrees
  Serial.println("Moving to 180 degrees (1600 steps)...");
  motorA.moveTo(1600);
  while (motorA.distanceToGo() != 0) motorA.run();
  delay(1000);

  // Move to 270 degrees
  Serial.println("Moving to 270 degrees (2400 steps)...");
  motorA.moveTo(2400);
  while (motorA.distanceToGo() != 0) motorA.run();
  delay(1000);

  // Move to 360 degrees
  Serial.println("Moving to 360 degrees (3200 steps)...");
  motorA.moveTo(3200);
  while (motorA.distanceToGo() != 0) motorA.run();
  delay(1000);

  // Move back to origin
  Serial.println("Sweeping back to 0 degrees...");
  motorA.moveTo(0);
  while (motorA.distanceToGo() != 0) motorA.run();

  Serial.println("Cycle complete. Waiting 2 seconds...");
  delay(2000);
}
