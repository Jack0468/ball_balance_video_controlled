import ok

def main():
    dev = ok.FrontPanelDevices().Open("")
    if not dev:
        print("Failed to open device")
        return

    pll = ok.PLL22393()
    dev.GetEepromPLL22393Configuration(pll)

    print(f"Reference Freq: {pll.GetReference()}")
    
    for i in range(3):
        p = pll.GetPLLP(i)
        q = pll.GetPLLQ(i)
        enable = pll.IsPLLEnabled(i)
        print(f"PLL {i}: P={p}, Q={q}, Enabled={enable}")

    for i in range(5):
        src = pll.GetOutputSource(i)
        div = pll.GetOutputDivider(i)
        enabled = pll.IsOutputEnabled(i)
        freq = pll.GetOutputFrequency(i)
        print(f"Output {i}: Source={src}, Divider={div}, Enabled={enabled}, Freq={freq}")

if __name__ == "__main__":
    main()
