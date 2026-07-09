#include <AccelStepper.h>

#define ENA_PIN PD0
#define STEP_PIN_A PD3
#define DIR_PIN_A PD2
#define STEP_PIN_B PD5
#define DIR_PIN_B PD4
#define STEP_PIN_C PD7
#define DIR_PIN_C PD6

AccelStepper motorA(1, STEP_PIN_A, DIR_PIN_A);
AccelStepper motorB(1, STEP_PIN_B, DIR_PIN_B);
AccelStepper motorC(1, STEP_PIN_C, DIR_PIN_C);

long stepsA = 0;
long stepsB = 0;
long stepsC = 0;
int step_size = 10;

void setup() {
  Serial.begin(115200);
  pinMode(ENA_PIN, OUTPUT);
  
  // ENA is active LOW for TMC2208
  digitalWrite(ENA_PIN, LOW); 
  
  motorA.setMaxSpeed(1000);
  motorA.setAcceleration(800);
  motorB.setMaxSpeed(1000);
  motorB.setAcceleration(800);
  motorC.setMaxSpeed(1000);
  motorC.setAcceleration(800);

  delay(2000);
  Serial.println("\n--- Platform Calibration Mode ---");
  Serial.println("Send commands via Serial Monitor to move the arms.");
  Serial.println(" q / a : Motor A Up / Down");
  Serial.println(" w / s : Motor B Up / Down");
  Serial.println(" e / d : Motor C Up / Down");
  Serial.println(" + / - : Change step size (Current: 10)");
}

void loop() {
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    bool moved = false;
    
    // Motor A
    if (cmd == 'q') { stepsA += step_size; motorA.moveTo(stepsA); moved = true; }
    else if (cmd == 'a') { stepsA -= step_size; motorA.moveTo(stepsA); moved = true; }
    
    // Motor B
    else if (cmd == 'w') { stepsB += step_size; motorB.moveTo(stepsB); moved = true; }
    else if (cmd == 's') { stepsB -= step_size; motorB.moveTo(stepsB); moved = true; }
    
    // Motor C
    else if (cmd == 'e') { stepsC += step_size; motorC.moveTo(stepsC); moved = true; }
    else if (cmd == 'd') { stepsC -= step_size; motorC.moveTo(stepsC); moved = true; }
    
    // Step Size
    else if (cmd == '+') { step_size *= 2; Serial.print("Step size: "); Serial.println(step_size); }
    else if (cmd == '-') { step_size = max(1, step_size / 2); Serial.print("Step size: "); Serial.println(step_size); }
    
    if (moved) {
      Serial.print("Current Absolute Steps -> A: "); Serial.print(stepsA);
      Serial.print("\tB: "); Serial.print(stepsB);
      Serial.print("\tC: "); Serial.println(stepsC);
    }
  }
  
  motorA.run();
  motorB.run();
  motorC.run();
}
