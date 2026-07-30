import cv2
import time

print("Opening camera 1...")
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Camera 1 could not be opened.")
else:
    print("Camera 1 opened successfully.")
    for i in range(5):
        ret, frame = cap.read()
        if ret:
            print(f"Read frame {i}: shape={frame.shape}")
        else:
            print(f"Failed to read frame {i}")
        time.sleep(0.5)
cap.release()
