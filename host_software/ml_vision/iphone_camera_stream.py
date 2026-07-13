import cv2
import socket
import time

# --- Configuration ---
# Hardcoded IP of the tethered laptop (e.g. standard iOS USB tethering IP is usually 172.20.10.2)
# Update this directly in Pyto if the laptop gets a different IP.
LAPTOP_IP = '172.20.10.2'
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

    # Optional: set camera resolution explicitly if supported
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_time = 1.0 / FPS_TARGET
    
    try:
        while True:
            start_t = time.time()
            
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break
                
            # Force resize to 640x480 just in case the camera ignored the property setting
            frame = cv2.resize(frame, (640, 480))
            
            # Compress to JPEG
            # Quality 80 provides a good balance between visual clarity and low network bandwidth
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
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
