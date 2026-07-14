import cv2
import socket
import time

# --- Configuration ---
# Hardcoded IP of the tethered laptop (e.g. standard iOS USB tethering IP is usually 172.20.10.2)
# Update this directly in Pyto if the laptop gets a different IP.
LAPTOP_IP = '172.20.10.4'
UDP_PORT = 5005
FPS_TARGET = 30
# ---------------------

def main():
    print(f"Starting UDP Video Stream to {LAPTOP_IP}:{UDP_PORT}")
    
    # Initialize UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    # Initialize iPhone Camera (0 is usually back camera, 1 is front)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open iPhone camera.")
        return

    # We avoid explicitly setting cv2.CAP_PROP_FRAME_WIDTH and HEIGHT here
    # as it causes an NSInvalidArgumentException on iOS when the dimension
    # exceeds the activeFormat's dimensions. The frame is resized in the loop.
    
    print("Camera opened, giving it a moment to warm up...")
    time.sleep(1.0)

    frame_time = 1.0 / FPS_TARGET
    
    try:
        while True:
            start_t = time.time()
            
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame, retrying...")
                time.sleep(0.1)
                continue
                
            # Force resize to 320x240 to ensure the JPEG fits inside a single UDP packet.
            # Our inference model expects 320x240 anyway, so this saves bandwidth.
            frame = cv2.resize(frame, (320, 240))
            
            # Compress to JPEG
            # Quality 60 to further guarantee a small payload size under the UDP MTU limit.
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 60]
            success, buffer = cv2.imencode('.jpg', frame, encode_param)
            
            if success:
                # Blast the byte array directly over UDP
                try:
                    sock.sendto(buffer.tobytes(), (LAPTOP_IP, UDP_PORT))
                except Exception as e:
                    print(f"Network error: {e}")
                    
            # Throttle to maintain target FPS
            elapsed = time.time() - start_t
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)

    except KeyboardInterrupt:
        print("\nStopping camera stream...")
    finally:
        cap.release()
        sock.close()
        print("Stream closed.")

if __name__ == '__main__':
    main()
