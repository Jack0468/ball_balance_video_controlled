"""
camera_calibration.py

Computes the camera matrix and distortion coefficients using a checkerboard pattern.
Includes a utility function to undistort individual frames.
"""

import cv2
import numpy as np
import glob
import os

class CameraCalibrator:
    def __init__(self, checkerboard_size=(9, 6)):
        self.checkerboard_size = checkerboard_size
        # termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
        
        # Arrays to store object points and image points from all the images.
        self.objpoints = [] # 3d point in real world space
        self.imgpoints = [] # 2d points in image plane.
        
        self.camera_matrix = None
        self.dist_coeffs = None

    def calibrate(self, images_path_pattern):
        images = glob.glob(images_path_pattern)
        if not images:
            print(f"No images found for pattern: {images_path_pattern}")
            return False

        gray = None
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)

            # If found, add object points, image points (after refining them)
            if ret:
                self.objpoints.append(self.objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                self.imgpoints.append(corners2)

        if len(self.objpoints) > 0 and gray is not None:
            ret, self.camera_matrix, self.dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                self.objpoints, self.imgpoints, gray.shape[::-1], None, None
            )
            print("Camera calibration successful.")
            return True
        else:
            print("Could not find checkerboard in images.")
            return False

    def get_calibration_data(self):
        return self.camera_matrix, self.dist_coeffs

    def undistort_frame(self, frame):
        if self.camera_matrix is None or self.dist_coeffs is None:
            print("Error: Camera not calibrated yet.")
            return frame
        
        h, w = frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeffs, (w,h), 1, (w,h))
        
        # undistort
        dst = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs, None, newcameramtx)
        
        # crop the image based on ROI if needed, but often we keep full size
        x, y, w_roi, h_roi = roi
        if w_roi > 0 and h_roi > 0:
             dst = dst[y:y+h_roi, x:x+w_roi]
        
        return dst

if __name__ == "__main__":
    # Example usage:
    # calibrator = CameraCalibrator(checkerboard_size=(9, 6))
    # calibrator.calibrate("data/calibration_images/*.jpg")
    # mtx, dist = calibrator.get_calibration_data()
    pass
