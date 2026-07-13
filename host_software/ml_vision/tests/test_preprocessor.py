
import sys
import os
# Add ml_vision root to sys.path to allow importing from core
_ml_vision_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ml_vision_root not in sys.path:
    sys.path.insert(0, _ml_vision_root)

"""
test_preprocessor.py

Tests the Canny edge detection and perspective warp logic.
Reads an image, applies the preprocessor, and saves the output.
"""

import cv2
import argparse
import os
from core.preprocessor import Preprocessor

def test_preprocess(image_path, output_path):
    if not os.path.exists(image_path):
        print(f"Error: Could not find image at {image_path}")
        return

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not read image at {image_path}")
        return
        
    print(f"Processing {image_path}...")
    
    # Initialize preprocessor (platform size 500x500)
    preproc = Preprocessor(platform_size=(500, 500))
    
    M, warped = preproc.get_perspective_transform(frame)
    
    if M is not None:
        print("Success! Perspective transform matrix calculated.")
        cv2.imwrite(output_path, warped)
        print(f"Warped image saved to {output_path}")
    else:
        print("Failed to detect 4-point contour for the platform.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to the input test image")
    parser.add_argument("output_path", help="Path to save the warped output image")
    args = parser.parse_args()
    
    test_preprocess(args.image_path, args.output_path)
