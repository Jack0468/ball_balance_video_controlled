import cv2
import cv2.aruco as aruco
import numpy as np

# Physical corners of the platform in mm (Top-Left origin)
# Used to calculate the Homography from pixels to millimeters
PLATFORM_W = 187.5
PLATFORM_H = 142.0

# Define where the 4 ArUco markers are placed physically on the board.
# We assume they are exactly in the corners:
# ID 0: Top-Left (0, 0)
# ID 1: Top-Right (PLATFORM_W, 0)
# ID 2: Bottom-Right (PLATFORM_W, PLATFORM_H)
# ID 3: Bottom-Left (0, PLATFORM_H)
PHYSICAL_CORNERS_MM = np.array([
    [0.0, 0.0],
    [PLATFORM_W, 0.0],
    [PLATFORM_W, PLATFORM_H],
    [0.0, PLATFORM_H]
], dtype=np.float32)

def main():
    print("Starting ArUco Webcam Tracker...")
    print("Press 'q' to quit.")
    
    cap = cv2.VideoCapture(0)
    
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
            
        if ids is not None and len(ids) == 4:
            # We found all 4 markers!
            # Sort corners by ID (0, 1, 2, 3)
            ids = ids.flatten()
            
            # Make sure we have exactly IDs 0, 1, 2, 3
            if set(ids) == {0, 1, 2, 3}:
                # Extract the center pixel of each marker
                pixel_centers = np.zeros((4, 2), dtype=np.float32)
                
                for i in range(4):
                    # Get index of marker with ID i
                    idx = np.where(ids == i)[0][0]
                    # corners[idx] is shape (1, 4, 2)
                    marker_corners = corners[idx][0]
                    # Center is average of 4 corners of the marker
                    center = np.mean(marker_corners, axis=0)
                    pixel_centers[i] = center
                    
                # Draw the detected markers and their centers
                aruco.drawDetectedMarkers(frame, corners, ids)
                for i, center in enumerate(pixel_centers):
                    cv2.circle(frame, tuple(center.astype(int)), 5, (0, 255, 0), -1)
                    
                # Calculate Homography Matrix (Pixels -> MM)
                M, status = cv2.findHomography(pixel_centers, PHYSICAL_CORNERS_MM)
                
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
