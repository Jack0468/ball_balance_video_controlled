import cv2
import numpy as np

def test_detect():
    cap = cv2.VideoCapture('host_software/ml_vision/data/01_bronze/video1/20260710_054604000_iOS.MOV')
    for _ in range(30):
        ret, frame = cap.read()
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    kernel = np.ones((5,5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    for cnt in contours[:5]:
        area = cv2.contourArea(cnt)
        if area > 10000:
            peri = cv2.arcLength(cnt, True)
            # Try different epsilon values
            for eps in [0.02, 0.05, 0.08, 0.1]:
                approx = cv2.approxPolyDP(cnt, eps * peri, True)
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(approx)
                    print(f"FOUND 4 CORNERS! eps={eps}, area={area}, bbox={(x,y,w,h)}")
                    return
            print(f"Failed for area={area}")

if __name__ == "__main__":
    test_detect()
