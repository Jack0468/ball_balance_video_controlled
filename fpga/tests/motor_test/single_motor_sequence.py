import ok
import time

def split_32(val):
    val = int(val) & 0xFFFFFFFF
    return val & 0xFFFF, (val >> 16) & 0xFFFF

def merge_16(lsb, msb):
    val = (msb << 16) | lsb
    if val & 0x80000000: 
        val -= 0x100000000
    return val

print("--- Hardware Motor Test: 90->180->270->360->0 ---")

dev = ok.FrontPanelDevices().Open("")
dev.ConfigureFPGA("motor_test_top.bit")

print("Configuring PLL22393 for robust hardware clock (clk1)...")
pll = ok.PLL22393()
pll.SetReference(48.0)
pll.SetPLLParameters(0, 48, 48, True)
pll.SetOutputSource(0, ok.PLL22393.ClkSrc_PLL0_0)
pll.SetOutputDivider(0, 1)
pll.SetOutputEnable(0, True)
dev.SetPLL22393Configuration(pll)

# NOTE: You must compile single_motor_test_top.v in Xilinx ISE first!
print("Loading FPGA Bitstream...")
dev.ConfigureFPGA("single_motor_test_top_old.bit")

fp = dev.GetFPGADataPortClassic() 

# 1. Zero the hardware step counters


# 800 steps = 90 degrees
# 1600 steps = 180 degrees
# 2400 steps = 270 degrees
# 3200 steps = 360 degrees
targets = [800, 1600, 2400, 3200, 0]


for target in targets:
    print(f"\nMoving to target: {target} steps...")
    lsb, msb = split_32(target)
    fp.SetWireInValue(0x01, lsb, 0xFFFF)
    fp.SetWireInValue(0x02, msb, 0xFFFF)
    fp.UpdateWireIns()
    
    # Latch the target
    fp.ActivateTriggerIn(0x40, 0)
    
    # Wait for physical motion (adjust delay if needed based on motor speed)
    time.sleep(2)
    
    fp.UpdateWireOuts()
    current_pos = merge_16(fp.GetWireOutValue(0x21), fp.GetWireOutValue(0x22))
    print(f"Hardware reported position: {current_pos}")

print("\nTest Sequence Complete!")
