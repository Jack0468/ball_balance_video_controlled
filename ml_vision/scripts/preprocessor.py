"""
preprocessor.py

Handles the preprocessing pipeline:
1. Undistort frames (optional, if calibration data is provided).
2. Adjust exposure/lighting.
3. Detect ArUco markers to find platform corners.
4. Perform perspective transformation to get a top-down view of the platform.
"""

import cv2
import numpy as np

class Preprocessor:
    def __init__(self, camera_matrix=None, dist_coeffs=None, platform_size=(500, 500)):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.platform_size = platform_size # (width, height) of the output warped image
        
        # Canny edge detection parameters
        self.canny_threshold1 = 50
        self.canny_threshold2 = 150
        
        # Desired corner coordinates in the output image (top-down view)
        # Order: top-left, top-right, bottom-right, bottom-left
        self.dst_pts = np.array([
            [0, 0],
            [self.platform_size[0] - 1, 0],
            [self.platform_size[0] - 1, self.platform_size[1] - 1],
            [0, self.platform_size[1] - 1]
        ], dtype="float32")

    def adjust_exposure(self, frame, alpha=1.0, beta=0):
        """
        Adjust exposure using simple contrast and brightness.
        alpha: Contrast control (1.0-3.0)
        beta: Brightness control (0-100)
        """
        adjusted = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        return adjusted

    def undistort(self, frame):
        if self.camera_matrix is not None and self.dist_coeffs is not None:
            h, w = frame.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeffs, (w,h), 1, (w,h))
            dst = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs, None, newcameramtx)
            return dst
        return frame

    def get_perspective_transform(self, frame):
        """
        Uses Canny edge detection to find the largest quadrilateral (assumed to be the platform)
        and returns the warped top-down view.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, self.canny_threshold1, self.canny_threshold2)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, frame

        # Find the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Approximate the contour to a polygon
        peri = cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)

        # If the largest contour has 4 corners, we assume it's the platform
        if len(approx) == 4:
            # Order points: top-left, top-right, bottom-right, bottom-left
            # We use a simple sorting method based on sums and differences of x,y
            pts = approx.reshape(4, 2)
            rect = np.zeros((4, 2), dtype="float32")
            
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)] # Top-Left
            rect[2] = pts[np.argmax(s)] # Bottom-Right
            
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)] # Top-Right
            rect[3] = pts[np.argmax(diff)] # Bottom-Left
            
            M = cv2.getPerspectiveTransform(rect, self.dst_pts)
            warped = cv2.warpPerspective(frame, M, self.platform_size)
            
            return M, warped

        return None, frame

    def process_frame(self, frame):
        """
        Full preprocessing pipeline.
        """
        frame_undistorted = self.undistort(frame)
        frame_adjusted = self.adjust_exposure(frame_undistorted)
        M, frame_warped = self.get_perspective_transform(frame_adjusted)
        
        return frame_warped if M is not None else frame_adjusted
