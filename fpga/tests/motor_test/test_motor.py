import ok 

# Open device 


dev = ok.FrontPanelDevices().Open("")
dev.ConfigureFPGA("motor_test_top.bit")
pll = ok.PLL22393()
pll.SetReference(48.0)                 # 48 MHz crystal reference on the XEM3010
pll.SetPLLParameters(0, 48, 48, True)  # PLL0 = 48 * 48/48 = 48 MHz (adjust to taste)
pll.SetOutputSource(0, ok.PLL22393.ClkSrc_PLL0_0)
pll.SetOutputDivider(0, 1)             # 48 MHz / 1 = 48 MHz on this output
pll.SetOutputEnable(0, True)
dev.SetPLL22393Configuration(pll)    
fp = dev.GetFPGADataPortClassic() 

N = 1

fp.SetWireInValue(0x00, N)   
fp.SetWireInValue(0x01, 0x0001); fp.UpdateWireIns()   # assert reset
fp.SetWireInValue(0x01, 0x0000); fp.UpdateWireIns()   # release -> running
fp.UpdateWireIns()

fp.UpdateWireOuts()
print(hex(fp.GetWireOutValue(0x20)))
