import cv2
import cv2.aruco as aruco
import numpy as np

# Physical corners of the platform in mm (Top-Left origin)
# Used to calculate the Homography from pixels to millimeters
PLATFORM_W = 187.5
PLATFORM_H = 142.0

# Define where the ArUco markers are placed physically on the board (in mm).
# The beauty of Homography is that they DO NOT have to be exactly in the corners!
# These are the exact millimeter coordinates of the centers of the 6 markers
# relative to the Top-Left (0,0) corner of the printed PDF bounding box.
MARKER_PHYSICAL_MM = {
    0: [12.0, 12.0],
    1: [175.5, 12.0],
    2: [175.5, 130.0],
    3: [12.0, 130.0],
    4: [12.0, 71.0],
    5: [175.5, 71.0]
}


def main():
    print("Starting ArUco Webcam Tracker...")
    print("Press 'q' to quit.")
    
    cap = cv2.VideoCapture(1)
    
    # Use 4x4 dictionary
    try:
        dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        parameters = aruco.DetectorParameters()
        detector = aruco.ArucoDetector(dictionary, parameters)
    except AttributeError:
        # Older OpenCV syntax fallback
        dictionary = aruco.Dictionary_get(aruco.DICT_4X4_50)
        parameters = aruco.DetectorParameters_create()
        detector = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect markers
        if detector is not None:
            corners, ids, rejected = detector.detectMarkers(gray)
        else:
            corners, ids, rejected = aruco.detectMarkers(gray, dictionary, parameters=parameters)
            
        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)
            ids = ids.flatten()
            
            # Match detected IDs to our physical dictionary
            pixel_centers = []
            physical_centers = []
            
            for i, marker_id in enumerate(ids):
                if marker_id in MARKER_PHYSICAL_MM:
                    # Get center of this specific marker
                    marker_corners = corners[i][0]
                    center = np.mean(marker_corners, axis=0)
                    
                    pixel_centers.append(center)
                    physical_centers.append(MARKER_PHYSICAL_MM[marker_id])
                    
                    # Draw center dot
                    cv2.circle(frame, tuple(center.astype(int)), 5, (0, 255, 0), -1)
            
            # We need at least 4 markers to compute a Homography
            if len(pixel_centers) >= 4:
                pixel_centers = np.array(pixel_centers, dtype=np.float32)
                physical_centers = np.array(physical_centers, dtype=np.float32)
                
                # cv2.findHomography uses RANSAC, so if we pass 5 or 6 redundant markers, 
                # it will optimally solve for the best matrix!
                M, status = cv2.findHomography(pixel_centers, physical_centers)
                
                if M is not None:
                    # Draw a virtual box on the center of the platform
                    center_mm = np.array([[[PLATFORM_W/2.0, PLATFORM_H/2.0]]], dtype=np.float32)
                    
                    # Convert M to inverse (MM -> Pixels) to project graphics onto the screen
                    M_inv = np.linalg.inv(M)
                    
                    # Project center back to pixels
                    center_px = cv2.perspectiveTransform(center_mm, M_inv)[0][0]
                    cv2.circle(frame, tuple(center_px.astype(int)), 10, (0, 0, 255), -1)
                    cv2.putText(frame, "CENTER", tuple(center_px.astype(int) + np.array([10, -10])), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                
                    cv2.putText(frame, "Homography: LOCKED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
        else:
            if ids is not None:
                aruco.drawDetectedMarkers(frame, corners, ids)
            cv2.putText(frame, f"Found {len(ids) if ids is not None else 0}/4 Markers", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        
        cv2.imshow("ArUco Tracker", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
