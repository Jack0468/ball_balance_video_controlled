"""
output_formatter.py

Formats the predicted (x, y) coordinates from the vision model into 
the output vector required by the inverse kinematics math or microcontroller.
"""

import json

class OutputFormatter:
    def __init__(self, platform_origin=(0, 0), scale_factor=1.0):
        """
        platform_origin: The (x, y) pixel coordinate of the platform's logical origin (0,0) in the real world.
                         Often the center of the platform.
        scale_factor: Conversion from pixels to real-world units (e.g., mm/pixel).
        """
        self.platform_origin = platform_origin
        self.scale_factor = scale_factor

    def format_output(self, pixel_coord):
        """
        Converts a pixel (x,y) to a real-world (x,y) vector and packages it.
        """
        if pixel_coord is None:
            return self._build_payload(None, None, valid=False)

        # Apply translation and scaling
        real_x = (pixel_coord[0] - self.platform_origin[0]) * self.scale_factor
        real_y = (pixel_coord[1] - self.platform_origin[1]) * self.scale_factor
        
        # Note: Depending on coordinate systems, y might need to be inverted.
        
        return self._build_payload(real_x, real_y, valid=True)

    def _build_payload(self, x, y, valid):
        """
        Constructs the final payload (e.g., JSON or byte array) to send to the MCU.
        """
        payload = {
            "valid": valid,
            "x": round(x, 3) if valid else 0.0,
            "y": round(y, 3) if valid else 0.0
        }
        # In a real scenario, this might be packed using `struct.pack` for UART transmission.
        return json.dumps(payload)

if __name__ == "__main__":
    # Example
    # formatter = OutputFormatter(platform_origin=(250, 250), scale_factor=0.5)
    # print(formatter.format_output((300, 200))) # Should print {"valid": true, "x": 25.0, "y": -25.0}
    pass
