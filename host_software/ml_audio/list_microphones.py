import sounddevice as sd


def main():
    print("\n--- Available Audio Input Devices ---\n")

    devices = sd.query_devices()
    default_input_idx = sd.default.device[0]

    for idx, device in enumerate(devices):
        # We only care about input devices (microphones)
        if device["max_input_channels"] > 0:
            is_default = idx == default_input_idx
            prefix = " [*] " if is_default else " [ ] "

            print(f"{prefix}Device ID {idx}: {device['name']}")
            print(f"      - Input Channels: {device['max_input_channels']}")
            print(f"      - Default Sample Rate: {device['default_samplerate']} Hz")
            print()

    print(
        "Note: The [*] indicates the default microphone that is currently being used by the scripts."
    )
    print(
        "If you need to change this, you can change your default microphone in Windows Sound Settings."
    )
    print("-------------------------------------\n")


if __name__ == "__main__":
    main()
