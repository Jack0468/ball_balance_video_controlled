#include <Arduino.h>
#include "MotorControl.h"

void setup() {
  Serial.begin(2000000);
  Serial.setTimeout(10);

  motor_init();
  home_motors();
  go_home();
  delay(1000);
}

void loop() {
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');

    int comma1 = cmd.indexOf(',');
    int comma2 = cmd.indexOf(',', comma1 + 1);

    if (comma1 > 0 && comma2 > 0) {
      long targetA = cmd.substring(0, comma1).toInt();
      long targetB = cmd.substring(comma1 + 1, comma2).toInt();
      long targetC = cmd.substring(comma2 + 1).toInt();

      pos[0] = targetA;
      pos[1] = targetB;
      pos[2] = targetC;

      motorA.moveTo(pos[0]);
      motorB.moveTo(pos[1]);
      motorC.moveTo(pos[2]);

      // Reply with CURRENT (lagged) positions — this is the actual motor
      // state the net trained on. Reading currentPosition() right here, before
      // run() chases the new target, gives the position reached from the
      // PREVIOUS command, which is exactly the lagged signal we want.
      Serial.print(motorA.currentPosition());
      Serial.print(',');
      Serial.print(motorB.currentPosition());
      Serial.print(',');
      Serial.println(motorC.currentPosition());
    }
  }

  speed_controller();
  motorA.run();
  motorB.run();
  motorC.run();
}