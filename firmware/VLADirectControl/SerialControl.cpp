#include "SerialControl.h"

unsigned long last_vla_packet_time = 0;

void check_vla_serial_commands(double &theta_a, double &theta_b, double &theta_c) {
    // 1. Failsafe logic:
    // If we lose connection to host PC for > 250ms, gently level out the platform
    // but keep motors engaged so it doesn't crash down.
    if (millis() - last_vla_packet_time > 250) {
        theta_a = 0;
        theta_b = 0;
        theta_c = 0;
    }
    
    // 2. Read Serial buffer
    // Expected Payload: '<' (1 byte) + theta_a (2 bytes) + theta_b (2 bytes) + theta_c (2 bytes) = 7 bytes total
    // Sending floats over serial is tricky, so host should send scaled int16_t (e.g. angle * 100)
    while (Serial.available() >= 7) {
        if (Serial.read() == '<') {
            int16_t raw_a = 0;
            int16_t raw_b = 0;
            int16_t raw_c = 0;
            
            Serial.readBytes((char*)&raw_a, 2);
            Serial.readBytes((char*)&raw_b, 2);
            Serial.readBytes((char*)&raw_c, 2);
            
            // Unscale
            theta_a = (double)raw_a / 100.0;
            theta_b = (double)raw_b / 100.0;
            theta_c = (double)raw_c / 100.0;
            
            last_vla_packet_time = millis();
        }
    }
}
