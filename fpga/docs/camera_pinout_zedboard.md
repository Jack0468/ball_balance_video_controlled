# OV7670 to ZedBoard Wiring Guide

This document provides the exact pin-to-pin wiring required to connect your OV7670 camera module to the ZedBoard, based on our `zedboard_ov7670.xdc` constraints file.

We are utilizing two PMOD headers on the ZedBoard: **PMOD JC** and **PMOD JD**. Both of these headers provide the required 3.3V logic levels.

> [!IMPORTANT]  
> **The Pixel Clock (PCLK) connection is critical!** It must be connected exactly to PMOD JC, Pin 1 (Zynq Pin AB7). This is a dedicated Multi-Region Clock Capable (MRCC) pin. If this is wired to a different pin, Vivado will throw fatal routing errors.

---

### PMOD JC Connections (Primary Control & Upper Data)

| OV7670 Pin | ZedBoard PMOD | Zynq Pin | Signal Description |
| :--- | :--- | :--- | :--- |
| **PCLK** | **JC1** | `AB7` | Pixel Clock (Input to FPGA). **CRITICAL: Must be this pin.** |
| **XCLK** | **JC2** | `AB6` | System Clock (24MHz Output from FPGA to Camera). |
| **VSYNC** | **JC3** | `Y4` | Vertical Sync (Indicates start of a new frame). |
| **HREF** | **JC4** | `AA4` | Horizontal Reference (Indicates active pixels in a row). |
| **GND** | **JC5** | `GND` | Ground |
| **3V3** | **JC6** | `3V3` | 3.3V Power |
| **SIOC** | **JC7** | `R6` | I2C / SCCB Clock (For configuring camera registers). |
| **SIOD** | **JC8** | `T6` | I2C / SCCB Data (For configuring camera registers). |
| **D7** | **JC9** | `T4` | Pixel Data Bit 7 (MSB) |
| **D6** | **JC10** | `U4` | Pixel Data Bit 6 |
| **GND** | **JC11** | `GND` | Ground |
| **3V3** | **JC12** | `3V3` | 3.3V Power |

---

### PMOD JD Connections (Lower Data)

| OV7670 Pin | ZedBoard PMOD | Zynq Pin | Signal Description |
| :--- | :--- | :--- | :--- |
| **D5** | **JD1** | `V7` | Pixel Data Bit 5 |
| **D4** | **JD2** | `W7` | Pixel Data Bit 4 |
| **D3** | **JD3** | `V5` | Pixel Data Bit 3 |
| **D2** | **JD4** | `V4` | Pixel Data Bit 2 |
| **GND** | **JD5** | `GND` | Ground |
| **3V3** | **JD6** | `3V3` | 3.3V Power |
| **D1** | **JD7** | `W6` | Pixel Data Bit 1 |
| **D0** | **JD8** | `W5` | Pixel Data Bit 0 (LSB) |
| **RESET** | *(Optional)* | - | Pull HIGH (3.3V) or leave disconnected (internally pulled up). |
| **PWDN** | *(Optional)* | - | Pull LOW (GND) or leave disconnected (internally pulled down). |

> [!TIP]
> Make sure to connect the Ground (`GND`) and Power (`3V3`) pins from the PMOD headers to the camera to ensure both share the exact same electrical reference.
