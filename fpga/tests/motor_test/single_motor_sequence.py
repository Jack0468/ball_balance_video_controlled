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

dev = ok.okCFrontPanel()
if dev.OpenBySerial("") != 0:
    print("FATAL: Opal Kelly XEM3010 device not found.")
    exit(1)
    
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
dev.ConfigureFPGA("single_motor_test_top.bit")

# 1. Zero the hardware step counters
print("Zeroing hardware step counters...")
dev.ActivateTriggerIn(0x40, 1) 
time.sleep(0.5)

# 800 steps = 90 degrees
# 1600 steps = 180 degrees
# 2400 steps = 270 degrees
# 3200 steps = 360 degrees
targets = [800, 1600, 2400, 3200, 0]

for target in targets:
    print(f"\nMoving to target: {target} steps...")
    lsb, msb = split_32(target)
    dev.SetWireInValue(0x01, lsb, 0xFFFF)
    dev.SetWireInValue(0x02, msb, 0xFFFF)
    dev.UpdateWireIns()
    
    # Latch the target
    dev.ActivateTriggerIn(0x40, 0)
    
    # Wait for physical motion (adjust delay if needed based on motor speed)
    time.sleep(2)
    
    dev.UpdateWireOuts()
    current_pos = merge_16(dev.GetWireOutValue(0x21), dev.GetWireOutValue(0x22))
    print(f"Hardware reported position: {current_pos}")

print("\nTest Sequence Complete!")
