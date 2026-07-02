"""
fpga_bridge.py

Skeleton for the Opal Kelly FrontPanel API.
Responsible for connecting to the FPGA and sending physical (x, y) coordinates.
Includes a Mock Mode since the physical board is not yet available.
"""

import time

try:
    import ok
    HAS_OK = True
except ImportError:
    HAS_OK = False
    print("Warning: Opal Kelly FrontPanel API (ok) not found. Running in MOCK mode.")

class MockFrontPanel:
    """A mock class that simulates Opal Kelly FrontPanel for testing without hardware."""
    def __init__(self):
        self.wires = {}
        
    def OpenBySerial(self, serial):
        print(f"[Mock FPGA] Connected to serial: {serial}")
        return 0 # 0 means success in OK API
        
    def ConfigureFPGA(self, bitfile):
        print(f"[Mock FPGA] Configured with bitfile: {bitfile}")
        return 0
        
    def SetWireInValue(self, ep_addr, val, mask):
        self.wires[ep_addr] = val & mask
        
    def UpdateWireIns(self):
        # In real API, this flushes the local wire states to the FPGA
        print(f"[Mock FPGA] Updated Wires: {self.wires}")

class FPGABridge:
    def __init__(self, bitfile_path=None, serial=""):
        if HAS_OK:
            self.dev = ok.okCFrontPanel()
        else:
            self.dev = MockFrontPanel()
            
        if self.dev.OpenBySerial(serial) != 0:
            raise RuntimeError("FPGA could not be opened.")
            
        if bitfile_path:
            if self.dev.ConfigureFPGA(bitfile_path) != 0:
                raise RuntimeError("FPGA configuration failed.")
                
        # Endpoint definitions (example addresses)
        self.EP_X_COORD = 0x00
        self.EP_Y_COORD = 0x01
        self.EP_STATE   = 0x02

    def float_to_fixed_wire(self, value):
        """
        Converts a floating point coordinate (e.g., mm) into a fixed-point integer
        representation compatible with ap_fixed<32, 16> on the FPGA.
        Assuming Q16.16 format for the WireIn.
        """
        # Multiply by 2^16 to shift fractional bits into integer range
        fixed_val = int(value * (1 << 16))
        # Mask to 32 bits just to be safe
        return fixed_val & 0xFFFFFFFF

    def send_coordinates(self, x_mm, y_mm, state=0):
        """
        Sends the physical coordinates and state to the FPGA.
        """
        x_fixed = self.float_to_fixed_wire(x_mm)
        y_fixed = self.float_to_fixed_wire(y_mm)
        
        self.dev.SetWireInValue(self.EP_X_COORD, x_fixed, 0xFFFFFFFF)
        self.dev.SetWireInValue(self.EP_Y_COORD, y_fixed, 0xFFFFFFFF)
        self.dev.SetWireInValue(self.EP_STATE, state, 0xFFFFFFFF)
        
        self.dev.UpdateWireIns()

if __name__ == "__main__":
    bridge = FPGABridge()
    print("Sending test coordinates...")
    bridge.send_coordinates(15.5, -10.2, state=1)
    time.sleep(0.1)
    bridge.send_coordinates(0.0, 0.0, state=0)
