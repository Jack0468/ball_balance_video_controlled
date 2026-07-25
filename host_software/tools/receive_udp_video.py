import socket
import struct
import numpy as np
import cv2
import time
import os

UDP_IP = "0.0.0.0" # Listen on all interfaces
UDP_PORT = 5001

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
        # Since the camera is unconfigured, it defaults to YUV422 (YUYV or UYVY).
        # Convert raw bytes to uint8 array
        img8 = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((HEIGHT, WIDTH, 2))
        
        # Try extracting just the Y (Luminance/Brightness) channel to get a grayscale image.
        # This is a foolproof way to see if we have valid camera data even if the colors are wrong.
        # In YUYV, Y is the 0th byte. In UYVY, Y is the 1st byte. Let's try 0th byte first.
        y_channel_0 = img8[:, :, 0]
        y_channel_1 = img8[:, :, 1]
        
        # Also try full color conversion (assuming YUYV)
        try:
            img_bgr = cv2.cvtColor(img8, cv2.COLOR_YUV2BGR_YUYV)
        except Exception as e:
            img_bgr = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        
        # Display!
        cv2.imshow("ZedBoard Live Camera Feed", img_bgr)
        cv2.imshow("Grayscale (Y0)", y_channel_0)
        cv2.imshow("Grayscale (Y1)", y_channel_1)
        cv2.waitKey(1)
            
    except Exception as e:
        print(f"Render error: {e}")

if __name__ == "__main__":
    main()
