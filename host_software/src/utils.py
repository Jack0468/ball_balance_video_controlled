import serial.tools.list_ports

def find_stm32_port():
    ports = serial.tools.list_ports.comports()
    if len(ports) == 1:
        return ports[0].device
    for p in ports:
        if "STMicroelectronics" in p.description or "STM" in p.description or "USB Serial" in p.description:
            return p.device
    return None
