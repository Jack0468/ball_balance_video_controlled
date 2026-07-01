#include <Wire.h>

#define OV7670_I2C_ADDRESS 0x21

// Sync & Clock Pins
#define XCLK_PIN 9   // PTC3
#define VSYNC_PIN 4  // PTA13
#define HREF_PIN 5   // PTD7
#define PCLK_PIN 6   // PTD4

// Data Pins (PORTC)
// D0: 10 (PTC4)
// D1: 13 (PTC5)
// D2: 11 (PTC6)
// D3: 12 (PTC7)
// D4: 35 (PTC8)
// D5: 36 (PTC9)
// D6: 37 (PTC10)
// D7: 38 (PTC11)

// Image Settings
#define QQVGA_WIDTH 160
#define QQVGA_HEIGHT 120
uint16_t frameBuffer[QQVGA_WIDTH * QQVGA_HEIGHT];

// Basic OV7670 QQVGA RGB565 Initialization Array
const uint8_t OV7670_REG_SETUP[][2] = {
  {0x12, 0x80}, // COM7: Reset registers
  {0xFF, 0xFF}, // Delay marker
  
  // Timing / Clock
  {0x11, 0x01}, // CLKRC: Internal clock pre-scaler (0x01 = Divide by 2 -> 30fps to avoid USB bottlenecking)
  {0x3b, 0x0a}, // COM11: Night mode, banding filter
  {0x3a, 0x04}, // TSLB: YUYV, UYVY formatting (very important for colors)

  // Output format (RGB565)
  {0x12, 0x04}, // COM7: Output format RGB (Color bar disabled)
  {0x40, 0x10}, // COM15: RGB565 output format
  {0x8c, 0x00}, // RGB444: Disable
  {0x3a, 0x04}, // TSLB: UYVY formatting

  // Resolution (QQVGA)
  {0x0c, 0x04}, // COM3: Enable scaling
  {0x3e, 0x1a}, // COM14: Scaling PCLK and manual scaling enable
  {0x72, 0x22}, // SCALING_DCWCTR
  {0x73, 0xf2}, // SCALING_PCLK_DIV
  {0x17, 0x16}, // HSTART
  {0x18, 0x04}, // HSTOP
  {0x32, 0xa4}, // HREF
  {0x19, 0x02}, // VSTART
  {0x1a, 0x7a}, // VSTOP
  {0x03, 0x0a}, // VREF

  // Color Matrix (crucial for RGB colors, otherwise it's just green/black)
  {0x4f, 0x80}, {0x50, 0x80}, {0x51, 0x00},
  {0x52, 0x22}, {0x53, 0x5e}, {0x54, 0x80},
  {0x56, 0x40}, {0x58, 0x9e}, {0x59, 0x88},
  {0x5a, 0x88}, {0x5b, 0x44}, {0x5c, 0x67},
  {0x5d, 0x49}, {0x5e, 0x0e}, {0x69, 0x00},
  {0x6a, 0x40}, {0x6b, 0x0a}, {0x6c, 0x0a},
  {0x6d, 0x55}, {0x6e, 0x11}, {0x6f, 0x9f},
  {0xb0, 0x84},
  
  // Image Quality Enhancements
  {0x41, 0x38}, // COM16: Enable Edge Enhancement, De-noise, and Color Matrix AWG

  // Auto Exposure / Auto Gain / Auto White Balance
  {0x13, 0xe7}, // COM8: Enable AEC, AGC, AWB
  {0x00, 0x00}, // GAIN
  {0x10, 0x00}, // AECH
  {0x0d, 0x40}, // COM4
  {0x14, 0x18}, // COM9: Automatic gain ceiling 8x
  {0xa5, 0x05}, {0xab, 0x07}, {0x24, 0x95}, {0x25, 0x33},
  {0x26, 0xe3}, {0x9f, 0x78}, {0xa0, 0x68}, {0xa1, 0x03},
  
  {0xFF, 0xFF}  // End marker
};

bool writeRegister(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(OV7670_I2C_ADDRESS);
  Wire.write(reg);
  Wire.write(val);
  return (Wire.endTransmission() == 0);
}

void initCamera() {
  Wire.begin();
  delay(100);
  
  for(int i = 0; ; i++) {
    uint8_t reg = OV7670_REG_SETUP[i][0];
    uint8_t val = OV7670_REG_SETUP[i][1];
    
    // Check for end marker
    if(reg == 0xFF && val == 0xFF) {
      if(i == 1) { 
        // Delay marker after reset
        delay(100);
        continue;
      } else {
        break; 
      }
    }
    
    if (!writeRegister(reg, val)) {
      Serial.print("ERR: I2C write failed at register 0x");
      Serial.println(reg, HEX);
      while(1) {
        delay(1000);
        Serial.println("ERR: I2C Init Failed. Is camera wired correctly?");
      }
    }
  }
}

void setup() {
  Serial.begin(115200);
  Serial.println("Starting OV7670 High-Speed Test...");
  
  // 1. Generate XCLK Hardware PWM (10 MHz on Pin 9)
  // On Teensy 3.5/3.6 (60MHz bus), 10MHz gives exactly 6 clock ticks.
  // 128/255 * 6 = exactly 3 ticks (perfect 50% duty cycle).
  pinMode(XCLK_PIN, OUTPUT);
  analogWriteFrequency(XCLK_PIN, 10000000); 
  analogWrite(XCLK_PIN, 128); // 50% duty cycle

  // 2. Configure Sync Pins
  pinMode(VSYNC_PIN, INPUT);
  pinMode(HREF_PIN, INPUT);
  pinMode(PCLK_PIN, INPUT);

  // 3. Configure Data Pins (PORTC 4-11)
  pinMode(10, INPUT);
  pinMode(13, INPUT);
  pinMode(11, INPUT);
  pinMode(12, INPUT);
  pinMode(35, INPUT);
  pinMode(36, INPUT);
  pinMode(37, INPUT);
  pinMode(38, INPUT);

  // 4. Send I2C Configurations
  initCamera();
  Serial.println("OV7670 Initialized over I2C.");
}

bool captureFrame() {
  uint32_t timeout;
  
  // Wait for VSYNC high (Start of new frame)
  timeout = micros();
  while(true) {
    if ((GPIOA_PDIR & (1 << 13))) { // VSYNC went HIGH
      delayMicroseconds(50); // Software debounce to ignore electrical noise
      if ((GPIOA_PDIR & (1 << 13))) break; // Still HIGH, it's a real frame!
    }
    if (micros() - timeout > 500000) { Serial.println("ERR: VSYNC stuck LOW"); return false; }
  }
  
  // Wait for VSYNC low (Active frame)
  timeout = micros();
  while(true) {
    if (!(GPIOA_PDIR & (1 << 13))) { // VSYNC went LOW
      delayMicroseconds(50); // Software debounce to ignore electrical noise
      if (!(GPIOA_PDIR & (1 << 13))) break; // Still LOW, data is starting!
    }
    if (micros() - timeout > 500000) { Serial.println("ERR: VSYNC stuck HIGH"); return false; }
  }
  
  uint32_t p = 0; // Pixel index buffer pointer
  
  // Disable all Teensy interrupts to ensure maximum speed and zero pixel loss
  noInterrupts(); 
  
  for(int y = 0; y < QQVGA_HEIGHT; y++) {
    
    // Wait for HREF high (Start of active row)
    // Add a small software counter timeout (micros() won't work with noInterrupts)
    uint32_t stuck = 0;
    while(!(GPIOD_PDIR & (1 << 7))) {
      stuck++;
      if (stuck > 1000000) { interrupts(); Serial.println("ERR: HREF stuck LOW"); return false; }
    }
    
    for(int x = 0; x < QQVGA_WIDTH; x++) {
      
      // ----------- BYTE 1 (High Byte) -----------
      stuck = 0;
      while((GPIOD_PDIR & (1 << 4))) { // Wait for PCLK low
        stuck++; if (stuck > 100000) { interrupts(); Serial.println("ERR: PCLK stuck HIGH"); return false; }
      }  
      stuck = 0;
      while(!(GPIOD_PDIR & (1 << 4))) { // Wait for PCLK high (data valid)
        stuck++; if (stuck > 100000) { interrupts(); Serial.println("ERR: PCLK stuck LOW"); return false; }
      } 
      // Read all 8 bits from PORTC and shift down to LSB
      uint8_t highByte = (GPIOC_PDIR >> 4) & 0xFF; 
      
      // ----------- BYTE 2 (Low Byte) -----------
      stuck = 0;
      while((GPIOD_PDIR & (1 << 4))) { // Wait for PCLK low
        stuck++; if (stuck > 100000) { interrupts(); Serial.println("ERR: PCLK stuck HIGH"); return false; }
      }  
      stuck = 0;
      while(!(GPIOD_PDIR & (1 << 4))) { // Wait for PCLK high (data valid)
        stuck++; if (stuck > 100000) { interrupts(); Serial.println("ERR: PCLK stuck LOW"); return false; }
      } 
      // Read all 8 bits from PORTC and shift down to LSB
      uint8_t lowByte = (GPIOC_PDIR >> 4) & 0xFF;
      
      // Assemble the 16-bit RGB565 pixel and store it
      frameBuffer[p++] = (highByte << 8) | lowByte;
    }
    
    // Wait for HREF low (End of active row)
    stuck = 0;
    while((GPIOD_PDIR & (1 << 7))) {
      stuck++;
      if (stuck > 1000000) { interrupts(); Serial.println("ERR: HREF stuck HIGH"); return false; }
    }
  }
  
  // Re-enable interrupts so the Teensy can process USB/Serial again
  interrupts(); 
  return true;
}

void loop() {
  if (captureFrame()) {
    // Stream the frame over USB Serial ONLY if successful
    // 1. Send a magic sync header so the Python script can align to the start of a frame
    // Expanded from 4 bytes to 8 bytes to prevent the "rolling film" desync issue!
    const uint8_t syncHeader[] = {0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x99, 0x88};
    Serial.write(syncHeader, 8);
    
    // 2. Blast the entire 38,400 byte buffer (160x120 * 2 bytes)
    Serial.write((uint8_t*)frameBuffer, sizeof(frameBuffer));
    
    // Force the USB driver to flush the buffer instantly
    Serial.send_now();
  } else {
    // If it timed out, pause for half a second so we can read the error message
    // without flooding the serial monitor
    delay(500); 
  }
}
