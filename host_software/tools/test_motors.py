import ok
import time

def split_32(val):
    """Splits a 32-bit signed int into two 16-bit unsigned shorts (LSB, MSB)"""
    val = int(val) & 0xFFFFFFFF
    return val & 0xFFFF, (val >> 16) & 0xFFFF

def merge_16(lsb, msb):
    """Merges two 16-bit unsigned shorts into a 32-bit signed int"""
    val = (msb << 16) | lsb
    if val & 0x80000000: 
        val -= 0x100000000
    return val

print("--- Hardware Motor Test for VRI_2026 ---")

# 1. Initialize Opal Kelly
dev = ok.okCFrontPanel()
if dev.OpenBySerial("") != 0:
    print("FATAL: Opal Kelly XEM3010 device not found.")
    exit(1)
    
print("Device opened successfully.")

print("Configuring PLL22393 for robust hardware clock (clk1)...")
pll = ok.PLL22393()
pll.SetReference(48.0)                 # 48 MHz crystal reference on the XEM3010
pll.SetPLLParameters(0, 48, 48, True)  # PLL0 = 48 MHz
pll.SetOutputSource(0, ok.PLL22393.ClkSrc_PLL0_0)
pll.SetOutputDivider(0, 1)             # 48 MHz / 1 = 48 MHz on clk1
pll.SetOutputEnable(0, True)
dev.SetPLL22393Configuration(pll)

# Note: In the future, this should point to the compiled bitstream
# dev.ConfigureFPGA("camera_usb_streamer.bit")

# 2. Zero the hardware step counters
print("Sending TriggerIn(0x40, bit 1) to zero all hardware step counters...")
dev.ActivateTriggerIn(0x40, 1) 
time.sleep(0.5)

# 3. Command motors to move 1 full revolution (3200 steps)
target_steps = 3200
lsb, msb = split_32(target_steps)

print(f"Loading {target_steps} steps into WireIn holding registers...")
# Motor 1
dev.SetWireInValue(0x01, lsb, 0xFFFF)
dev.SetWireInValue(0x02, msb, 0xFFFF)
# Motor 2
dev.SetWireInValue(0x03, lsb, 0xFFFF)
dev.SetWireInValue(0x04, msb, 0xFFFF)
# Motor 3
dev.SetWireInValue(0x05, lsb, 0xFFFF)
dev.SetWireInValue(0x06, msb, 0xFFFF)

dev.UpdateWireIns()

# 4. Fire the atomic latch trigger
print("Firing TriggerIn(0x40, bit 0) to execute atomic motion...")
dev.ActivateTriggerIn(0x40, 0)

# 5. Wait for the hardware P-controller to finish the physical motion
print("Waiting 3 seconds for physical movement...")
time.sleep(3)

# 6. Read back exact current positions from the FPGA step counters
dev.UpdateWireOuts()
pos1 = merge_16(dev.GetWireOutValue(0x21), dev.GetWireOutValue(0x22))
pos2 = merge_16(dev.GetWireOutValue(0x23), dev.GetWireOutValue(0x24))
pos3 = merge_16(dev.GetWireOutValue(0x25), dev.GetWireOutValue(0x26))

print("\n--- Final Hardware Step Counters ---")
print(f"Motor 1: {pos1}")
print(f"Motor 2: {pos2}")
print(f"Motor 3: {pos3}")

if pos1 == target_steps and pos2 == target_steps and pos3 == target_steps:
    print("SUCCESS: Hardware PID successfully reached target positions.")
else:
    print("WARNING: Hardware PID did not reach targets. Check if speeds are clamped or time was insufficient.")
