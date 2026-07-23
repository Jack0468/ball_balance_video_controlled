"""
coordinate_math.py

Translates 2D pixel coordinates from the vision pipeline into physical 
real-world coordinates (millimeters) for the FPGA's Inverse Kinematics.
"""

class PixelToPhysicalMapper:
    def __init__(self, pixel_width, pixel_height, physical_width_mm, physical_height_mm):
        """
        Initializes the mapper.
        
        Args:
            pixel_width: Width of the warped image in pixels (e.g., 500)
            pixel_height: Height of the warped image in pixels (e.g., 500)
            physical_width_mm: Real-world width of the platform in mm
            physical_height_mm: Real-world height of the platform in mm
        """
        self.pixel_width = pixel_width
        self.pixel_height = pixel_height
        self.physical_width_mm = physical_width_mm
        self.physical_height_mm = physical_height_mm
        
        # Scaling factors (mm per pixel)
        self.scale_x = physical_width_mm / pixel_width
        self.scale_y = physical_height_mm / pixel_height
        
        # Center coordinates in pixels
        self.center_pixel_x = pixel_width / 2.0
        self.center_pixel_y = pixel_height / 2.0

    def pixels_to_mm(self, px, py):
        """
        Converts a (px, py) pixel coordinate into a physical (mm_x, mm_y) 
        offset from the exact center of the platform.
        """
        # Offset from center in pixels
        offset_x_px = px - self.center_pixel_x
        offset_y_px = py - self.center_pixel_y
        
        # Convert to mm
        mm_x = offset_x_px * self.scale_x
        mm_y = offset_y_px * self.scale_y
        
        return mm_x, mm_y

if __name__ == "__main__":
    # Simple unit test
    mapper = PixelToPhysicalMapper(500, 500, 300.0, 300.0)
    
    # Test center (should be 0, 0)
    print(f"Center (250, 250) -> {mapper.pixels_to_mm(250, 250)} mm")
    
    # Test top-left (should be -150, -150)
    print(f"Top-Left (0, 0) -> {mapper.pixels_to_mm(0, 0)} mm")
    
    # Test bottom-right (should be 150, 150)
    print(f"Bottom-Right (500, 500) -> {mapper.pixels_to_mm(500, 500)} mm")

import cv2
import numpy as np

class HomographyProjector:
    def __init__(self, dst_pts_mm):
        """
        Initializes the homography projector with the physical destination points.
        
        Args:
            dst_pts_mm (np.ndarray): Physical corners of the platform in mm.
                                     Typically a 4x2 array of [x, y].
        """
        self.dst_pts = np.array(dst_pts_mm, dtype=np.float32)
        self.M = None
        
    def update_homography(self, src_pts_px):
        """
        Updates the internal Homography matrix given 4 pixel coordinates.
        
        Args:
            src_pts_px (list or np.ndarray): 4 corner points in pixel space [x, y].
            
        Returns:
            bool: True if homography was successfully calculated, False otherwise.
        """
        src = np.array(src_pts_px, dtype=np.float32)
        if len(src) != 4:
            return False
            
        M, status = cv2.findHomography(src, self.dst_pts)
        if M is not None:
            self.M = M
            return True
        return False
        
    def project_point(self, px, py):
        """
        Projects a 2D pixel coordinate into physical mm space using the active Homography.
        
        Args:
            px (float): X pixel coordinate
            py (float): Y pixel coordinate
            
        Returns:
            tuple: (mm_x, mm_y) or (None, None) if Homography isn't set.
        """
        if self.M is None:
            return None, None
            
        pt = np.array([[[px, py]]], dtype=np.float32)
        mm_pt = cv2.perspectiveTransform(pt, self.M)
        return float(mm_pt[0][0][0]), float(mm_pt[0][0][1])
