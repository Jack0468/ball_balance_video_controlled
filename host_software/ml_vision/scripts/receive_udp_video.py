import socket
import struct
import numpy as np
import cv2
import time
import os

UDP_IP = "0.0.0.0" # Listen on all interfaces
UDP_PORT = 8080

WIDTH = 640
HEIGHT = 480
PIXEL_BYTES = 2
FRAME_SIZE = WIDTH * HEIGHT * PIXEL_BYTES
PACKET_PAYLOAD = 1024
PACKETS_PER_FRAME = (FRAME_SIZE + PACKET_PAYLOAD - 1) // PACKET_PAYLOAD

def main():
    print(f"Starting UDP Video Receiver on Port {UDP_PORT}...")
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # Increase socket buffer size to prevent dropping packets at OS level
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024 * 5) 
    sock.bind((UDP_IP, UDP_PORT))
    
    print("Waiting for ZedBoard connection...")
    
    current_frame_id = -1
    frame_buffer = bytearray(FRAME_SIZE)
    packets_received = 0
    
    frame_count = 0
    start_time = time.time()

    while True:
        try:
            data, addr = sock.recvfrom(2048)
            
            if len(data) < 8:
                continue
                
            # Header is two 32-bit little-endian unsigned ints: [frame_id, packet_id]
            frame_id, packet_id = struct.unpack("<II", data[:8])
            payload = data[8:]
            
            if frame_id != current_frame_id:
                # We received the first packet of a NEW frame.
                # Render the previous frame if we received most of its packets.
                if current_frame_id != -1 and packets_received > PACKETS_PER_FRAME * 0.8:
                    render_frame(frame_buffer)
                    
                    frame_count += 1
                    if frame_count % 30 == 0:
                        fps = 30 / (time.time() - start_time)
                        print(f"Live FPS: {fps:.1f} (Dropped packets this frame: {PACKETS_PER_FRAME - packets_received})")
                        start_time = time.time()
                
                # Setup state for the new frame
                current_frame_id = frame_id
                packets_received = 0
                
            # Copy payload into the correct byte offset in our frame buffer
            offset = packet_id * PACKET_PAYLOAD
            length = len(payload)
            
            if offset + length <= FRAME_SIZE:
                frame_buffer[offset:offset+length] = payload
                packets_received += 1
                
        except KeyboardInterrupt:
            print("\nShutting down receiver...")
            break
            
    cv2.destroyAllWindows()
    sock.close()

def render_frame(frame_bytes):
    try:
        # 1. Convert raw bytes to 16-bit array
        img16 = np.frombuffer(frame_bytes, dtype=np.uint16).reshape((HEIGHT, WIDTH))
        
        # 2. Extract RGB565 channels
        r = (img16 >> 11) & 0x1F
        g = (img16 >> 5) & 0x3F
        b = img16 & 0x1F
        
        # 3. Scale to 8-bit (0-255)
        r = (r * 255) // 31
        g = (g * 255) // 63
        b = (b * 255) // 31
        
        # 4. Stack for OpenCV (BGR order)
        img_bgr = np.dstack((b, g, r)).astype(np.uint8)
        
        # 5. Display!
        cv2.imshow("ZedBoard Live Camera Feed", img_bgr)
        cv2.waitKey(1)
            
    except Exception as e:
        print(f"Render error: {e}")

if __name__ == "__main__":
    main()
