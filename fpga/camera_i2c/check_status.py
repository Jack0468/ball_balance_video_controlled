import ok
import sys
import time

def main():
    dev = ok.FrontPanelDevices().Open("")
    if not dev:
        print("Failed to open FPGA device. It might be locked by another script.")
        sys.exit(1)

    fp = dev.GetFPGADataPortClassic()
    
    # Read counters over 1 second to see if they are ticking
    print("Reading camera debugging counters...")
    fp.UpdateWireOuts()
    pclk1 = fp.GetWireOutValue(0x26)
    href1 = fp.GetWireOutValue(0x25)
    vsync1 = fp.GetWireOutValue(0x24)
    conf = fp.GetWireOutValue(0x20)
    sdram_stat = fp.GetWireOutValue(0x21)
    overflow = fp.GetWireOutValue(0x23)
    
    time.sleep(0.5)
    
    fp.UpdateWireOuts()
    pclk2 = fp.GetWireOutValue(0x26)
    href2 = fp.GetWireOutValue(0x25)
    vsync2 = fp.GetWireOutValue(0x24)
    
    print(f"Configuration Done:    {conf & 1}")
    print(f"SDRAM Init Complete:   {(sdram_stat >> 1) & 1}")
    print(f"Frame Captured:        {sdram_stat & 1}")
    print(f"Overflow Sticky:       {overflow & 1}")
    print(f"PCLK count changed:    {pclk2 != pclk1}  (Val1={pclk1}, Val2={pclk2})")
    print(f"HREF count changed:    {href2 != href1}  (Val1={href1}, Val2={href2})")
    print(f"VSYNC count changed:   {vsync2 != vsync1}  (Val1={vsync1}, Val2={vsync2})")

if __name__ == "__main__":
    main()
