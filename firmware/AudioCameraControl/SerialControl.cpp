#include "SerialControl.h"

unsigned long last_packet_time = 0;

void check_serial_commands(double &target_x, double &target_y) {
    // 1. Failsafe logic:
    // If we lose connection for >100ms, gently level out the platform
    // but keep motors engaged so it doesn't crash down.
    if (millis() - last_packet_time > 100) {
        target_x = 0;
        target_y = 0;
    }
    
    // 2. Read Serial buffer
    // Expected Payload: '<' (1 byte) + error_x (2 bytes) + error_y (2 bytes) = 5 bytes total
    while (Serial.available() >= 5) {
        if (Serial.read() == '<') {
            int16_t err_x = 0;
            int16_t err_y = 0;
            
            // We have exactly the bytes we need in the buffer, so readBytes is safe
            Serial.readBytes((char*)&err_x, 2);
            Serial.readBytes((char*)&err_y, 2);
            
            target_x = err_x;
            target_y = err_y;
            
            last_packet_time = millis();
        }
    }
}
