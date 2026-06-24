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
        
        # ArUco dictionary
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
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
        Detects ArUco markers and returns the warped top-down view of the platform.
        Assumes 4 markers with IDs 0, 1, 2, 3 placed at TL, TR, BR, BL corners.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.aruco_detector.detectMarkers(gray)

        if ids is None or len(ids) < 4:
            # Cannot perform perspective transform if we don't see all 4 corners
            return None, frame

        # Map marker IDs to expected corner positions
        marker_centers = {}
        for i, marker_id in enumerate(ids.flatten()):
            # Calculate the center of the marker
            c = corners[i][0]
            center_x = int(np.mean(c[:, 0]))
            center_y = int(np.mean(c[:, 1]))
            marker_centers[marker_id] = (center_x, center_y)

        # Check if all required IDs are present
        required_ids = [0, 1, 2, 3] # TL, TR, BR, BL
        if not all(rid in marker_centers for rid in required_ids):
             return None, frame

        # Extract source points in the correct order
        src_pts = np.array([
            marker_centers[0],
            marker_centers[1],
            marker_centers[2],
            marker_centers[3]
        ], dtype="float32")

        # Compute the perspective transform matrix
        M = cv2.getPerspectiveTransform(src_pts, self.dst_pts)

        # Warp the image
        warped = cv2.warpPerspective(frame, M, self.platform_size)

        return M, warped

    def process_frame(self, frame):
        """
        Full preprocessing pipeline.
        """
        frame_undistorted = self.undistort(frame)
        frame_adjusted = self.adjust_exposure(frame_undistorted)
        M, frame_warped = self.get_perspective_transform(frame_adjusted)
        
        return frame_warped if M is not None else frame_adjusted
