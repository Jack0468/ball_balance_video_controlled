#include "SerialControl.h"

unsigned long last_packet_time = 0;

void check_serial_commands(double &cam_x, double &cam_y, bool &cam_active) {
    // 1. Force camera to ALWAYS be active (No failsafe)
    cam_active = true;
    
    // 2. Read Serial buffer
    // Expected Payload: '<' (1 byte) + cam_x (2 bytes) + cam_y (2 bytes) = 5 bytes total
    while (Serial.available() >= 5) {
        if (Serial.read() == '<') {
            int16_t in_x = 0;
            int16_t in_y = 0;
            
            Serial.readBytes((char*)&in_x, 2);
            Serial.readBytes((char*)&in_y, 2);
            
            cam_x = in_x;
            cam_y = in_y;
            
            last_packet_time = millis();
            cam_active = true;
        }
    }
}
