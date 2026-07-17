import ok
import numpy as np
import time
import cv2
import sys
import os

WI_START = 0x02
WI_RESET = 0x03
WO_FRAME = 0x21
WO_CONF  = 0x20
PIPE_OUT = 0xA0

BLOCK_SIZE  = 1024
W, H        = 640, 480
FRAME_BYTES = W * H * 2

def main():
    dev = ok.FrontPanelDevices().Open("")
    if not dev:
        print("Failed to open FPGA device.")
        sys.exit(1)

    print("FPGA Device opened. Configuring...")
    # Assume we run this from VRI_2026/fpga/camera_i2c
    dev.ConfigureFPGA("bitfiles/camera_top.bit")

    pll = ok.PLL22393()
    dev.GetEepromPLL22393Configuration(pll)
    pll.SetReference(48.0)

    # Only one PLL output is needed: clk1 (100MHz) for SDRAM on FPGA pin N9.
    # The camera XCLK (24MHz) is now generated INSIDE the FPGA via DCM_SP
    # dividing 100MHz by 25/6, forwarded to pin K5 via ODDR2.
    # We no longer route any PLL output to P9 (clk2 port removed from design).
    pll.SetPLLParameters(1, 25, 3, True)      # VCO = 48*(25/3) = 400MHz
    pll.SetOutputSource(1, ok.PLL22393.ClkSrc_PLL1_0)
    pll.SetOutputDivider(1, 4)                # 400MHz / 4 = 100MHz
    pll.SetOutputEnable(1, True)

    dev.SetPLL22393Configuration(pll)
    # Wait for the CY22393 VCOs and outputs to fully lock and stabilize before asserting reset
    time.sleep(0.1) 

    fp = dev.GetFPGADataPortClassic()
   
    fp.SetWireInValue(WI_RESET, 0x1)
    fp.UpdateWireIns()
    time.sleep(0.01)
    fp.SetWireInValue(WI_RESET, 0x0)
    fp.UpdateWireIns()
    time.sleep(0.05)

    fp.SetWireInValue(WI_START, 0x1)
    fp.UpdateWireIns()
    time.sleep(0.01)
    fp.SetWireInValue(WI_START, 0x0)
    fp.UpdateWireIns()

    t0 = time.time()
    while True:
        fp.UpdateWireOuts()
        if fp.GetWireOutValue(WO_CONF) & 0x1:
            print("Configuration done.", flush=True)
            break
        if time.time() - t0 > 2.0:
            print("Configuration timeout!", flush=True)
            break
    
    fp.SetWireInValue(WI_RESET, 0x1)
    fp.UpdateWireIns()
    time.sleep(0.01)
    fp.SetWireInValue(WI_RESET, 0x0)    
    fp.UpdateWireIns()
    
    WI_ARM = 0x04
    buf = bytearray(FRAME_BYTES)
    
    # Wait for SDRAM to initialize (takes ~40 clock cycles after reset deasserts, which we gave it in line 77)
    fp.UpdateWireOuts()
    sdram_stat = fp.GetWireOutValue(WO_FRAME)
    sdram_init = (sdram_stat >> 1) & 1
    if not sdram_init:
        print("ERROR: SDRAM failed to initialize! Controller might be stuck.", flush=True)
        # We'll continue anyway to see if it works, but this is a red flag
    else:
        print("SDRAM Initialized successfully.", flush=True)
    
    # Grab 5 frames to let it stabilize, then save the 5th
    frames_captured = 0
    max_frames = 5
    
    dev.SetTimeout(2000) # 2 second timeout
    while frames_captured < max_frames:
        fp.SetWireInValue(WI_ARM, 0x1)
        fp.UpdateWireIns()
        fp.SetWireInValue(WI_ARM, 0x0)
        fp.UpdateWireIns()

        # Poll WO_FRAME to wait for the SDRAM to capture a full frame
        timeout = time.time() + 2.0
        frame_ready = False
        while time.time() < timeout:
            fp.UpdateWireOuts()
            if fp.GetWireOutValue(WO_FRAME) & 0x1:
                frame_ready = True
                break
            time.sleep(0.01)
            
        if not frame_ready:
            print("Timeout waiting for frame to buffer in SDRAM!", flush=True)
            fp.SetWireInValue(WI_RESET, 0x01)
            fp.UpdateWireIns()
            fp.SetWireInValue(WI_RESET, 0x00)
            fp.UpdateWireIns()
            time.sleep(0.05)
            continue

        n = fp.ReadFromPipeOut(PIPE_OUT, buf)
        
        if n != FRAME_BYTES:
            print(f"Short read: {n} bytes.", flush=True)
            fp.SetWireInValue(WI_RESET, 0x1)
            fp.UpdateWireIns()
            fp.SetWireInValue(WI_RESET, 0x0)
            fp.UpdateWireIns()
            time.sleep(0.05)
            continue
    
        fp.UpdateWireOuts()
        if fp.GetWireOutValue(0x23) & 0x1:
            print("OVERFLOW!")
            fp.SetWireInValue(WI_RESET, 0x1)
            fp.UpdateWireIns()
            fp.SetWireInValue(WI_RESET, 0x0)
            fp.UpdateWireIns()
            continue

        frames_captured += 1
        print(f"Captured frame {frames_captured}")

    # Process the last frame
    px = np.frombuffer(buf, dtype='<u2').reshape(H, W)
    r = ((px >> 11) & 0x1F) << 3
    g = ((px >> 5)  & 0x3F) << 2
    b = ( px        & 0x1F) << 3
    bgr = np.dstack((b, g, r)).astype(np.uint8)
    
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_frame.png")
    cv2.imwrite(out_path, bgr)
    print(f"Saved image to {out_path}")

if __name__ == "__main__":
    main()
