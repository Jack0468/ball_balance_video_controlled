"""
classical_cv_model.py

A baseline model using OpenCV for detecting the ball using color masking
and Hough Circles. Assumes the image has already been perspective warped.
"""

import cv2
import numpy as np

class ClassicalCVModel:
    def __init__(self, color_lower=(30, 150, 50), color_upper=(255, 255, 180)):
        # Default HSV color bounds (e.g., for a bright ball)
        self.color_lower = np.array(color_lower, dtype="uint8")
        self.color_upper = np.array(color_upper, dtype="uint8")

    def predict(self, frame):
        """
        Detects the ball and returns its (x, y) coordinates relative to the top-down frame.
        """
        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Apply color mask
        mask = cv2.inRange(hsv, self.color_lower, self.color_upper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        center = None
        if len(contours) > 0:
            # Find the largest contour
            c = max(contours, key=cv2.contourArea)
            
            # Compute the bounding circle
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            
            # Calculate moments to find the center
            M = cv2.moments(c)
            if M["m00"] > 0:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            else:
                center = (int(x), int(y))
                
            # Only return if it's large enough to be the ball
            if radius > 5:
                return center
                
        # Fallback to Hough Circles if contour fails
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                   param1=50, param2=30, minRadius=5, maxRadius=50)
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            # Return the first found circle center
            return (circles[0, 0, 0], circles[0, 0, 1])

        return None
