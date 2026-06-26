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
        
        # Temporal smoothing state
        self.prev_corners = None
        # Increased from 0.2 to 0.8 to reduce visual lag/sluggishness on the display
        self.ema_alpha = 0.9
        
        # HSV Color masking parameters (White/Grey platform)
        self.hsv_lower = np.array([0, 0, 80])
        self.hsv_upper = np.array([180, 40, 255])

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

    def get_perspective_transform(self, frame, draw_corners=False):
        """
        Uses adaptive Canny edge detection to find the largest quadrilateral (assumed to be the platform),
        applies morphological closing, robustly extracts 4 corners (falling back to minAreaRect),
        and returns the warped top-down view with EMA smoothing.
        """
        # Convert to HSV and apply color mask (Targeting White/Grey)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        
        # Morphological operations to clean up the mask (remove salt & pepper noise)
        kernel_mask = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_mask)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_mask)
        
        # Generate an edge-map of the binary color mask using morphological gradient
        # This gives us a 1-pixel thin outline of every object the color filter caught.
        kernel_grad = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask_edges = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel_grad)
        
        # Show the mask in a window so the user can physically see what the color filter is catching
        try:
            cv2.imshow("HSV Tuning Mask", mask)
        except Exception:
            pass
        
        # Find contours directly on the binary mask instead of using Canny!
        # This completely ignores background clutter because everything not matching the color is already gone.
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, frame

        # Sort contours by area descending (look at the top 5 largest)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        
        h, w = frame.shape[:2]
        
        # 2. Smart Platform Extraction
        pts = None
        for c in contours:
            area = cv2.contourArea(c)
            # Ignore tiny contours that are definitely not the platform
            if area < 5000:
                continue
                
            peri = cv2.arcLength(c, True)
            # Relax the approximation slightly (0.03 instead of 0.02) to handle slightly rounded corners
            approx = cv2.approxPolyDP(c, 0.03 * peri, True)
            
            # The platform MUST have 4 corners and generally be a convex shape
            if len(approx) == 4 and cv2.isContourConvex(approx):
                # Check if any of the 4 corners are touching the physical edges of the camera frame
                # (This prevents the script from thinking the border of the image is the platform)
                touches_border = False
                for pt in approx:
                    px, py = pt[0]
                    if px <= 10 or py <= 10 or px >= w - 10 or py >= h - 10:
                        touches_border = True
                        break
                
                if touches_border:
                    continue
                    
                # Edge Overlap Verification
                # We draw the 4 edges of the candidate polygon and check how much they overlap with the actual color boundary
                poly_outline = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(poly_outline, [approx], -1, 255, 2)  # 2 pixel thickness
                
                overlap = cv2.bitwise_and(poly_outline, mask_edges)
                overlap_ratio = np.count_nonzero(overlap) / (np.count_nonzero(poly_outline) + 1e-6)
                
                # If less than 60% of the polygon's perimeter aligns with a real color boundary, reject it
                if overlap_ratio < 0.6:
                    continue
                    
                # Color Uniformity Check
                # Create a mask of just this quadrilateral and measure the variance/texture of the colors inside it
                poly_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(poly_mask, [np.int32(approx)], 255)
                mean_color, stddev_color = cv2.meanStdDev(frame, mask=poly_mask)
                avg_stddev = np.mean(stddev_color)
                
                # A solid colored platform (even with lighting gradients or small markers) will have low variance (< 45).
                # A highly textured object (like a poster or keyboard) will have high variance (> 60).
                if avg_stddev > 50.0:  # Adjust this threshold if it rejects the real platform!
                    continue
                
                pts = approx.reshape(4, 2)
                break
                
        # If we couldn't find a perfect 4-sided polygon, fallback to a bounding box around the absolute largest object
        if pts is None:
            largest_contour = contours[0]
            rect_min = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect_min)
            pts = np.int0(box) if hasattr(np, 'int0') else np.int32(box)

        if pts is not None and len(pts) == 4:
            # Order points: top-left, top-right, bottom-right, bottom-left
            rect = np.zeros((4, 2), dtype="float32")
            
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)] # Top-Left
            rect[2] = pts[np.argmax(s)] # Bottom-Right
            
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)] # Top-Right
            rect[3] = pts[np.argmax(diff)] # Bottom-Left
            
            # 3. Temporal Smoothing (EMA)
            if self.prev_corners is not None:
                dist = np.linalg.norm(rect - self.prev_corners)
                if dist < 150: # Threshold for valid EMA smoothing
                    rect = self.ema_alpha * rect + (1 - self.ema_alpha) * self.prev_corners
                
            self.prev_corners = rect.copy()
            
            if draw_corners:
                # Draw the outline of the platform
                cv2.polylines(frame, [np.int32(rect)], True, (255, 0, 0), 3)
                # Draw circles on the corners
                for i, pt in enumerate(rect):
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 6, (0, 0, 255), -1)
                    cv2.putText(frame, str(i), (int(pt[0])+10, int(pt[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
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
