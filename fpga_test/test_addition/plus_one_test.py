import ok

dev = ok.FrontPanelDevices().Open("")
if dev is None:
    raise RuntimeError("board not found")

if dev.ConfigureFPGA("top.bit") != 0:
    raise RuntimeError("FPGA configuration failed")
if not dev.IsFrontPanelEnabled():
    raise RuntimeError("FrontPanel not enabled")

fp = dev.GetFPGADataPortClassic()    # wire/pipe interface in 5.x

for test in [0, 1, 100, 1234, 65534]:
    fp.SetWireInValue(0x00, test)
    fp.UpdateWireIns()

    fp.UpdateWireOuts()
    result = fp.GetWireOutValue(0x20)

    print(f"sent {test:5d}  ->  got back {result:5d}   {'OK' if result == test + 1 else 'MISMATCH'}")