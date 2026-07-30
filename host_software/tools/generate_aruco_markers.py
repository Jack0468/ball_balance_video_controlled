import cv2
import cv2.aruco as aruco
import numpy as np
import os

def create_printable_sheet(output_path="aruco_markers_sheet.png", marker_size_px=400):
    """
    Generates 4 ArUco markers from the 4X4_50 dictionary and places them on a single
    white canvas so they can be easily printed.
    """
    # Use 4x4 dictionary for maximum robustness at low resolutions
    try:
        # OpenCV 4.7+ syntax
        dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    except AttributeError:
        # Older OpenCV syntax
        dictionary = aruco.Dictionary_get(aruco.DICT_4X4_50)
        
    markers = []
    # Generate markers with IDs 0, 1, 2, 3
    for i in range(4):
        try:
            # OpenCV 4.7+
            marker = aruco.generateImageMarker(dictionary, i, marker_size_px)
        except AttributeError:
            # Older OpenCV
            marker = aruco.drawMarker(dictionary, i, marker_size_px)
            
        # Add a white border around each marker (quiet zone)
        border_size = marker_size_px // 4
        marker_with_border = cv2.copyMakeBorder(
            marker, 
            border_size, border_size, border_size, border_size, 
            cv2.BORDER_CONSTANT, value=[255, 255, 255]
        )
        
        # Add label text to tell the user which corner it belongs to
        labels = [
            "ID: 0 (Top-Left)", 
            "ID: 1 (Top-Right)", 
            "ID: 2 (Bottom-Right)", 
            "ID: 3 (Bottom-Left)"
        ]
        
        # Convert to BGR so we can put colored text if we want, or just black text
        marker_bgr = cv2.cvtColor(marker_with_border, cv2.COLOR_GRAY2BGR)
        cv2.putText(marker_bgr, labels[i], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        
        markers.append(marker_bgr)
        
    # Arrange in a 2x2 grid for printing
    top_row = np.hstack((markers[0], markers[1]))
    bottom_row = np.hstack((markers[3], markers[2])) # 3 is Bottom-Left, 2 is Bottom-Right
    
    sheet = np.vstack((top_row, bottom_row))
    
    # Save the image
    cv2.imwrite(output_path, sheet)
    print(f"Successfully generated ArUco sheet: {os.path.abspath(output_path)}")
    print("INSTRUCTIONS:")
    print("1. Print this image without scaling (100% scale).")
    print("2. Measure the printed black squares to ensure they match your expected physical size (e.g., 30mm x 30mm).")
    print("3. Cut them out leaving a small white border.")
    print("4. Tape them to the 4 corners of your physical platform according to the labels.")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, "..", "data", "aruco")
    os.makedirs(out_dir, exist_ok=True)
    
    out_file = os.path.join(out_dir, "aruco_markers_sheet.png")
    create_printable_sheet(out_file)
