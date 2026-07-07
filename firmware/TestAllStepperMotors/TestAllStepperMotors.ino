#include <AccelStepper.h>

#define ENA_PIN 0

// Motor A pins
#define STEP_PIN_A 3
#define DIR_PIN_A 2

// Motor B pins
#define STEP_PIN_B 5
#define DIR_PIN_B 4

// Motor C pins
#define STEP_PIN_C 7
#define DIR_PIN_C 6

AccelStepper motorA(1, STEP_PIN_A, DIR_PIN_A);
AccelStepper motorB(1, STEP_PIN_B, DIR_PIN_B);
AccelStepper motorC(1, STEP_PIN_C, DIR_PIN_C);

// 3200 steps = 360 degrees
// 30 degrees = 3200 * (30 / 360) = 267 steps

void setup() {
  Serial.begin(115200);
  Serial.println("--- Stepper Motors 30 Degrees Test ---");

  pinMode(ENA_PIN, OUTPUT);
  digitalWrite(ENA_PIN, LOW);

  // Configure Motor A
  motorA.setMinPulseWidth(20);
  motorA.setMaxSpeed(1000);
  motorA.setAcceleration(500);

  // Configure Motor B
  motorB.setMinPulseWidth(20);
  motorB.setMaxSpeed(1000);
  motorB.setAcceleration(500);

  // Configure Motor C
  motorC.setMinPulseWidth(20);
  motorC.setMaxSpeed(1000);
  motorC.setAcceleration(500);
}

void loop() {
  // --- Test Motor A ---
  Serial.println("\nMotor A: Moving to 30 degrees (267 steps)...");
  motorA.moveTo(267);
  while (motorA.distanceToGo() != 0) motorA.run();
  delay(1000);

  Serial.println("Motor A: Returning to 0 degrees...");
  motorA.moveTo(0);
  while (motorA.distanceToGo() != 0) motorA.run();
  delay(1000);

  // --- Test Motor B ---
  Serial.println("\nMotor B: Moving to 30 degrees (267 steps)...");
  motorB.moveTo(267);
  while (motorB.distanceToGo() != 0) motorB.run();
  delay(1000);

  Serial.println("Motor B: Returning to 0 degrees...");
  motorB.moveTo(0);
  while (motorB.distanceToGo() != 0) motorB.run();
  delay(1000);

  // --- Test Motor C ---
  Serial.println("\nMotor C: Moving to 30 degrees (267 steps)...");
  motorC.moveTo(267);
  while (motorC.distanceToGo() != 0) motorC.run();
  delay(1000);

  Serial.println("Motor C: Returning to 0 degrees...");
  motorC.moveTo(0);
  while (motorC.distanceToGo() != 0) motorC.run();
  
  Serial.println("\nCycle complete. Waiting 2 seconds before repeating...");
  delay(2000);
}
