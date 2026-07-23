import cv2
import numpy as np

def analyze_color_bars(image_path="test_frame.png"):
    img = cv2.imread(image_path)
    if img is None:
        print("Could not find test_frame.png")
        return

    # Count unique colors
    # Reshape image into a 1D array of pixels (each pixel is 3 BGR values)
    pixels = img.reshape(-1, 3)
    unique_colors = np.unique(pixels, axis=0)
    
    print(f"Total Unique Colors Found: {len(unique_colors)}")
    
    if len(unique_colors) > 100:
        print("\n[RESULT] FALSE ALARM: The OV7670 Color Bar did NOT activate!")
        print("We found hundreds of unique colors. This means the camera is still outputting raw analog sensor noise.")
        print("To fix this, we likely need to set COM17 (Register 0x42) bit 3 to enable the DSP color bar, in addition to COM7.")
    elif len(unique_colors) <= 8:
        print("\n[RESULT] SUCCESS: Perfect Color Bars captured!")
        print("Your digital pipeline is 100% flawless. No dropped bytes, no shifted bits.")
    else:
        print("\n[RESULT] PIPELINE CORRUPTION DETECTED!")
        print("The color bar activated (hence less than 100 colors), but the bytes are scrambled!")
        print("This means the FPGA is stitching the High Byte of one pixel to the Low Byte of the next pixel, creating 'frankenstein' colors.")
        
        # Print the first 20 pixels of row 0 to see the repeating pattern
        print("\nFirst 20 pixels of Row 0 (BGR):")
        print(img[0, :20].tolist())

if __name__ == "__main__":
    analyze_color_bars()
