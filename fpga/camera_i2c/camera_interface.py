import ok
import numpy as np
import time
import cv2

WI_START = 0x02
WI_RESET = 0x03
WO_FRAME = 0x21
WO_CONF  = 0x22
PIPE_OUT = 0xA0     # okBTPipeOut

BLOCK_SIZE  = 1024            # bytes per BTPipe block (512 x 16-bit pixels)

W, H        = 640, 480 #640, 480
FRAME_BYTES = W * H * 2

if __name__ == "__main__":
    dev = ok.FrontPanelDevices().Open("")

    dev.ConfigureFPGA("bitfiles/camera_top.bit")

    # 24 MHz xclk for the OV7670
    pll = ok.PLL22393()
    pll.SetReference(48.0)
    pll.SetPLLParameters(0, 48, 48, True)
    pll.SetOutputSource(0, ok.PLL22393.ClkSrc_PLL0_0)
    pll.SetOutputDivider(0, 2) # 48/2 = 24 MHz
    pll.SetOutputEnable(0, True)
    dev.SetPLL22393Configuration(pll)
    time.sleep(0.05) 

    fp = dev.GetFPGADataPortClassic()
   
    fp.SetWireInValue(WI_START, 0x1)
    fp.UpdateWireIns()
    time.sleep(0.01)
    fp.SetWireInValue(WI_START, 0x0)
    fp.UpdateWireIns()

    t0 = time.time()
    while True:
        fp.UpdateWireOuts()
        if fp.GetWireOutValue(WO_CONF) & 0x1:
            print("configuration done")
            break
    
    fp.SetWireInValue(WI_RESET, 0x1)
    fp.UpdateWireIns()
    time.sleep(0.01)
    fp.SetWireInValue(WI_RESET, 0x0)    
    fp.UpdateWireIns()
    
    WI_ARM = 0x04
    buf = bytearray(FRAME_BYTES)
    
    while True:

        # repeatedly ask for a frame
        fp.SetWireInValue(WI_ARM, 0x1)
        fp.UpdateWireIns()
        fp.SetWireInValue(WI_ARM, 0x0)
        fp.UpdateWireIns()

        print("full " + str(fp.GetWireOutValue(0x21)))
        print("empty " + str(fp.GetWireOutValue(0x22)))
        print("overflow " + str(fp.GetWireOutValue(0x23)))

        # n = fp.ReadFromBlockPipeOut(PIPE_OUT, BLOCK_SIZE, buf)
        
        # if n != FRAME_BYTES:
        #     print(f"Short read: {n} bytes. Camera/USB desynced.")
        #     # Only fall back to a hard reset if the pipeline completely stalls
        #     fp.SetWireInValue(WI_RESET, 0x1)
        #     fp.UpdateWireIns()
        #     fp.SetWireInValue(WI_RESET, 0x0)
        #     fp.UpdateWireIns()
        #     time.sleep(0.05)
        #     continue
    
        fp.UpdateWireOuts()
        if fp.GetWireOutValue(0x23) & 0x1:
            print("OVERFLOW! USB bottlenecked and dropped pixels. Frame corrupted.")
            # Clear the sticky overflow state for the next run
            fp.SetWireInValue(WI_RESET, 0x1)
            fp.UpdateWireIns()
            fp.SetWireInValue(WI_RESET, 0x0)
            fp.UpdateWireIns()
            continue

        # # reset fifo repeatedly, each frame. this avoids relying on perfect synchronisation 
        # fp.SetWireInValue(WI_RESET, 0x1)
        # fp.UpdateWireIns()
        # fp.SetWireInValue(WI_RESET, 0x0)
        # fp.UpdateWireIns()

        # n = fp.ReadFromBlockPipeOut(PIPE_OUT, BLOCK_SIZE, buf)
        
        # if n != FRAME_BYTES:
        #     print(f"short/failed read: {n} bytes. Resynchronizing...")
        #     # Only reset if we actually dropped sync
        #     fp.SetWireInValue(WI_RESET, 0x1)
        #     fp.UpdateWireIns()
        #     fp.SetWireInValue(WI_RESET, 0x0)
        #     fp.UpdateWireIns()
        #     time.sleep(0.05)
        #     continue
    

        # fp.UpdateWireOuts()
        # if fp.GetWireOutValue(0x23) & 0x1:
        #     print("OVERFLOW !!!!!")
        #     continue  #skip if overflow          
    
        px = np.frombuffer(buf, dtype='<u2').reshape(H, W)
        r = ((px >> 11) & 0x1F) << 3
        g = ((px >> 5)  & 0x3F) << 2
        b = ( px        & 0x1F) << 3
        bgr = np.dstack((b, g, r)).astype(np.uint8)   # note order below
        cv2.imshow("OV7670", bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

