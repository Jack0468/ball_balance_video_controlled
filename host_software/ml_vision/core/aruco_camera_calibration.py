"""
aruco_camera_calibration.py

Computes the camera focal length and intrinsic matrix using ArUco markers 
defined in a manifest file (e.g., ground_truth_manifest.json).
Provides both standard multi-image calibration and a direct pinhole estimation 
method if the exact Z-distance to the platform is known.
"""

import cv2
import numpy as np
import json
import glob
import argparse

class ArucoCameraCalibrator:
    def __init__(self, manifest_path, marker_size_mm=None, dict_type=cv2.aruco.DICT_4X4_50):
        """
        Initializes the calibrator using the physical marker coordinates from the manifest.
        
        Args:
            manifest_path (str): Path to ground_truth_manifest.json
            marker_size_mm (float, optional): The exact physical size of the ArUco black square in mm. 
                                              If None, it tries to infer it from the manifest (which might be wrong if manifest has white border).
            dict_type: OpenCV ArUco dictionary type.
        """
        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)
            
        self.marker_size_mm = marker_size_mm
        
        try:
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
            self.aruco_params = cv2.aruco.DetectorParameters()
        except AttributeError:
            # Older OpenCV versions
            self.aruco_dict = cv2.aruco.Dictionary_get(dict_type)
            self.aruco_params = cv2.aruco.DetectorParameters_create()
            
        
        # Build 3D object points for each marker
        self.obj_points = {} # marker_id -> np.array of 4 corners (3D)
        
        for marker in self.manifest['aruco_markers']:
            m_id = marker['id']
            center_x = marker['center_mm'][0]
            center_y = marker['center_mm'][1]
            
            # Use provided black square size, or fallback to manifest size (which might include white border)
            size = self.marker_size_mm if self.marker_size_mm is not None else marker['size_mm']
            
            half_s = size / 2.0
            
            # cv2.aruco returns corners in order: top-left, top-right, bottom-right, bottom-left
            # Assuming manifest Y increases downward, X increases rightward
            top_left = [center_x - half_s, center_y - half_s, 0.0]
            top_right = [center_x + half_s, center_y - half_s, 0.0]
            bottom_right = [center_x + half_s, center_y + half_s, 0.0]
            bottom_left = [center_x - half_s, center_y + half_s, 0.0]
            
            self.obj_points[m_id] = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)

        self.camera_matrix = None
        self.dist_coeffs = None
        
    def _detect_markers(self, gray):
        try:
            # OpenCV 4.7+
            detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            corners, ids, rejected = detector.detectMarkers(gray)
        except AttributeError:
            # Older OpenCV
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        return corners, ids

    def calibrate(self, images_path_pattern):
        """
        Attempts standard intrinsic calibration using multiple images.
        Note: If all images are taken perfectly parallel at the exact same height,
        this mathematical model may be ill-conditioned. In that case, use 
        estimate_focal_length_from_known_z() instead.
        """
        images = glob.glob(images_path_pattern)
        if not images:
            print(f"No images found for pattern: {images_path_pattern}")
            return False

        all_corners = []
        all_ids = []
        image_size = None

        for fname in images:
            img = cv2.imread(fname)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if image_size is None:
                image_size = gray.shape[::-1]

            corners, ids = self._detect_markers(gray)
            
            if ids is not None and len(ids) > 0:
                all_corners.append(corners)
                all_ids.append(ids)

        if len(all_corners) == 0:
            print("Could not find any ArUco markers in the images.")
            return False

        # Prepare object points and image points for cv2.calibrateCamera
        objpoints_list = []
        imgpoints_list = []

        for i in range(len(all_corners)):
            frame_corners = all_corners[i]
            frame_ids = all_ids[i].flatten()
            
            frame_obj_pts = []
            frame_img_pts = []
            
            for j in range(len(frame_ids)):
                m_id = frame_ids[j]
                if m_id in self.obj_points:
                    frame_obj_pts.append(self.obj_points[m_id])
                    frame_img_pts.append(frame_corners[j][0]) 
            
            if len(frame_obj_pts) > 0:
                # concatenate all points in this frame to (N*4, 3) and (N*4, 2)
                objpoints_list.append(np.vstack(frame_obj_pts))
                imgpoints_list.append(np.vstack(frame_img_pts))

        if len(objpoints_list) > 0:
            ret, self.camera_matrix, self.dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                objpoints_list, imgpoints_list, image_size, None, None
            )
            print("--- Standard Multi-Image Calibration ---")
            print(f"RMS Reprojection Error: {ret:.4f} pixels")
            print(f"Focal Length: fx={self.camera_matrix[0,0]:.2f}, fy={self.camera_matrix[1,1]:.2f}")
            print("----------------------------------------\n")
            return True
        else:
            print("No matching markers found between manifest and images.")
            return False

    def estimate_focal_length_from_known_z(self, images_path_pattern, z_distance_mm):
        """
        Direct pinhole camera calculation.
        Use this if the camera is perfectly parallel to the platform at a known Z distance.
        It averages the computed focal length across all valid detected markers in all images.
        """
        images = glob.glob(images_path_pattern)
        if not images:
            print(f"No images found for pattern: {images_path_pattern}")
            return None
            
        focal_lengths_x = []
        focal_lengths_y = []
        
        for fname in images:
            img = cv2.imread(fname)
            if img is None:
                continue
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            corners, ids = self._detect_markers(gray)
            
            if ids is None or len(ids) == 0:
                continue
                
            for i in range(len(ids)):
                m_id = ids[i][0]
                if m_id in self.obj_points:
                    img_pts = corners[i][0] # (4, 2) top-left, top-right, bottom-right, bottom-left
                    obj_pts = self.obj_points[m_id] # (4, 3)
                    
                    # Compute pixel width/height of the marker
                    w_px = np.linalg.norm(img_pts[1] - img_pts[0])
                    h_px = np.linalg.norm(img_pts[3] - img_pts[0])
                    
                    # Compute physical width/height of the marker
                    w_mm = np.linalg.norm(obj_pts[1] - obj_pts[0])
                    h_mm = np.linalg.norm(obj_pts[3] - obj_pts[0])
                    
                    # Pinhole formula: f = (size_px / size_mm) * Z
                    fx = (w_px / w_mm) * z_distance_mm
                    fy = (h_px / h_mm) * z_distance_mm
                    
                    focal_lengths_x.append(fx)
                    focal_lengths_y.append(fy)
                
        if focal_lengths_x:
            avg_fx = np.mean(focal_lengths_x)
            avg_fy = np.mean(focal_lengths_y)
            print("--- Known Z-Distance Direct Estimation ---")
            print(f"Number of marker samples used: {len(focal_lengths_x)}")
            print(f"Focal Length at Z={z_distance_mm}mm: fx={avg_fx:.2f}, fy={avg_fy:.2f}")
            print("------------------------------------------")
            return avg_fx, avg_fy
        else:
            print("Could not estimate focal length; no matching markers found.")
            return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate camera focal length using ArUco markers.")
    parser.add_argument("--manifest", type=str, required=True, help="Path to ground_truth_manifest.json")
    parser.add_argument("--images", type=str, required=True, help="Glob pattern for calibration images (e.g. 'data/*.jpg')")
    parser.add_argument("--marker-size", type=float, default=15.0, help="The exact physical size of the printed ArUco black square in mm (e.g., 14.0 or 15.0).")
    parser.add_argument("--known-z", type=float, default=None, help="If provided, calculates focal length directly using this known camera-to-platform distance in mm.")
    
    args = parser.parse_args()
    
    calibrator = ArucoCameraCalibrator(args.manifest, marker_size_mm=args.marker_size)
    
    print(f"Loaded manifest: {args.manifest}")
    print(f"Using black square size: {args.marker_size} mm")
    print(f"Image pattern: {args.images}")
    print("")

    # 1. Attempt standard multi-image calibration
    calibrator.calibrate(args.images)
    
    # 2. If known-z is provided, attempt direct pinhole estimation
    if args.known_z is not None:
        calibrator.estimate_focal_length_from_known_z(args.images, args.known_z)
