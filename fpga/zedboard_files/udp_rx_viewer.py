import socket
import struct
import numpy as np
import cv2

# Configuration
UDP_IP = "0.0.0.0"      # Listen on all available interfaces
UDP_PORT = 5005         # Must match the port the ZedBoard is sending to
WIDTH = 640
HEIGHT = 480
BYTES_PER_PIXEL = 2     # We padded the 12-bit RGB444 to 16 bits (2 bytes)

# Initialize UDP Socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"Listening for OV7670 video stream on UDP port {UDP_PORT}...")

# Frame buffer to hold the 640x480 image
frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

current_row = 0

while True:
    try:
        # Receive a UDP packet containing exactly 1 row (1280 bytes)
        # We set buffer slightly larger to be safe
        data, addr = sock.recvfrom(2048) 
        
        if len(data) != WIDTH * BYTES_PER_PIXEL:
            print(f"Warning: Received unexpected packet size: {len(data)} bytes")
            continue

        # Unpack the 16-bit integers
        format_string = f"<{WIDTH}H" # Assuming Little-Endian transmission
        pixels = struct.unpack(format_string, data)

        # Reconstruct the row
        row_data = np.zeros((WIDTH, 3), dtype=np.uint8)
        
        for i in range(WIDTH):
            pix = pixels[i]
            # The format of pixel data is {RRRR GGGG BBBB}[cite: 1]
            r_4bit = (pix >> 8) & 0x0F
            g_4bit = (pix >> 4) & 0x0F
            b_4bit = pix & 0x0F
            
            # Scale 4-bit color (0-15) up to 8-bit color (0-255) by multiplying by 17
            row_data[i, 2] = r_4bit * 17  # OpenCV uses BGR by default
            row_data[i, 1] = g_4bit * 17
            row_data[i, 0] = b_4bit * 17
            
        # Place row into the frame buffer
        frame[current_row, :, :] = row_data
        
        current_row += 1
        
        # When we reach the bottom of the frame, display it and reset
        if current_row >= HEIGHT:
            cv2.imshow("ZedBoard OV7670 UDP Stream", frame)
            current_row = 0
            
            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        break

sock.close()
cv2.destroyAllWindows()