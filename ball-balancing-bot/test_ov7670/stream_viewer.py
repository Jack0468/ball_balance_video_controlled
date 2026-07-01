import serial
import numpy as np
import cv2
import sys

# Image specifications
WIDTH = 160
HEIGHT = 120
BYTES_PER_PIXEL = 2
FRAME_SIZE = WIDTH * HEIGHT * BYTES_PER_PIXEL
SYNC_HEADER = b'\xAA\xBB\xCC\xDD'

def main(port_name):
    print(f"Connecting to Teensy on {port_name}...")
    try:
        # Baud rate is ignored by Teensy USB, but required by PySerial
        ser = serial.Serial(port_name, 115200, timeout=1) 
        
        # Force DTR and RTS to reboot the Teensy and wake up the USB CDC driver
        import time
        ser.setDTR(False)
        ser.setRTS(False)
        time.sleep(0.1)
        ser.setDTR(True)
        ser.setRTS(True)
        time.sleep(0.5)
        
    except Exception as e:
        print(f"Failed to open port {port_name}: {e}")
        return

    print("Connected! Waiting for frames...")
    
    buffer = b''

    while True:
        try:
            # Read whatever is available (or block until at least 1 byte)
            waiting = max(1, ser.in_waiting)
            chunk = ser.read(waiting)
            if not chunk:
                print(".", end="", flush=True) # Print a dot for timeout (no data)
                continue
                
            # --- DEBUG: Print exactly what we received ---
            # If it's ascii text, it will look like b'ERR: VSYNC...'
            # If it's binary, it will look like b'\x00\x1f...'
            print(f"\n[DEBUG] Received {len(chunk)} bytes: {repr(chunk[:60])}")
            
            buffer += chunk
            
            # Look for the sync header
            idx = buffer.find(SYNC_HEADER)
            
            if idx != -1:
                # We found the header! Check if we have a full frame's worth of data yet.
                if len(buffer) >= idx + 4 + FRAME_SIZE:
                    # Extract the full frame
                    frame_data = buffer[idx+4 : idx+4+FRAME_SIZE]
                    # Keep any remaining bytes in the buffer for the next frame
                    buffer = buffer[idx+4+FRAME_SIZE:]
                else:
                    # We found the header but don't have all the pixels yet.
                    # Keep looping to read more data.
                    continue
            else:
                # Header not found. We drop the garbage data to prevent memory leaks.
                # However, let's first check if the Teensy is sending ASCII error messages!
                try:
                    text = buffer.decode('ascii', errors='ignore')
                    if "ERR" in text or "OV7670" in text:
                        # Print only the latest unique message to avoid spamming the console
                        lines = [line for line in text.split('\n') if line.strip()]
                        if lines:
                            print(f"Teensy says: {lines[-1].strip()}")
                except:
                    pass
                    
                # Keep the last 3 bytes just in case the 4-byte header was split across chunks.
                if len(buffer) > 3:
                    buffer = buffer[-3:]
                continue

            # 3. Convert raw bytes to RGB565 numpy array
            # The Teensy sends uint16_t which is 2 bytes per pixel
            # Numpy uses '<u2' for little-endian 16-bit unsigned integers
            raw_pixels = np.frombuffer(frame_data, dtype='<u2').reshape((HEIGHT, WIDTH))
            
            # 4. Decode RGB565 to RGB888 (24-bit color) for OpenCV
            # Extract 5-bit Red, 6-bit Green, 5-bit Blue
            r = ((raw_pixels >> 11) & 0x1F).astype(np.uint8)
            g = ((raw_pixels >> 5) & 0x3F).astype(np.uint8)
            b = (raw_pixels & 0x1F).astype(np.uint8)

            # Scale colors to full 0-255 range
            r = (r * 255) // 31
            g = (g * 255) // 63
            b = (b * 255) // 31

            # Stack into an OpenCV BGR image (OpenCV uses BGR, not RGB)
            bgr_image = np.dstack((b, g, r))

            # Upscale the 160x120 image so it's easier to see on modern monitors
            display_image = cv2.resize(bgr_image, (640, 480), interpolation=cv2.INTER_NEAREST)

            # 5. Display the frame
            cv2.imshow("OV7670 Real-Time Feed", display_image)
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error reading stream: {e}")
            break

    ser.close()
    cv2.destroyAllWindows()
    print("Stream closed.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python stream_viewer.py <COM_PORT>")
        print("Example: python stream_viewer.py COM3")
    else:
        main(sys.argv[1])
