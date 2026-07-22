# UDP Networking on Zynq-7000: Troubleshooting Findings

This document summarizes the investigation and resolution of the bare-metal UDP communication failure on the ZedBoard (Zynq-7000) using the lwIP network stack in Vitis.

## 1. The Problem

The goal was to establish a UDP video stream between the ZedBoard (running a bare-metal C application) and a host PC. During the initial tests, the host PC was unable to communicate with the ZedBoard:

* **Symptom:** Pinging the ZedBoard (`192.168.1.10`) from the host PC (`192.168.1.100`) resulted in `Destination host unreachable`.
* **Execution State:** Pausing the debugger revealed that the application was running perfectly inside the `while(1)` polling loop and was not frozen or hanging on AXI/VDMA transactions.
* **Network State:** The PC's `arp -a` table showed no entry for the ZedBoard. The Ethernet physical link was verified to be up, meaning the PHY auto-negotiation succeeded, but the ZedBoard was ignoring ARP broadcast requests.

## 2. Hardware-Level Verification (First Principles)

To ensure the Vivado hardware design (the bitstream) was not the root cause, we inspected the generated `ps7_init.c` to verify the Processing System (PS) initialization sequence:

1. **Ethernet Clock:** We verified that register `0xF8000140` (`ENET_CLK_CTRL`) was properly written, confirming that `ENET0_PERIPH_CLK` was active and the MAC hardware block was receiving clocks.
2. **MIO Pin Multiplexing:** We checked registers `0xF8000740` to `0xF800076C` (MIO pins 16 to 27) and confirmed they were correctly configured for Ethernet TX/RX (e.g., L2_SEL set to Ethernet).

**Conclusion:** The Vivado block design and hardware bitstream were 100% correct. The MAC was alive and physically receiving packets from the PHY.

## 3. Software-Level Root Cause Analysis

Because the hardware was verified and the program was actively polling `xemacif_input(&server_netif)` in the main loop without crashing, the issue pointed to a failure in the software interrupt pipeline.

### How packets are received in Xilinx lwIP:
In `NO_SYS=1` bare-metal environments, the MAC uses Direct Memory Access (DMA). When an Ethernet frame physically arrives:
1. The MAC pushes it into the RX DMA ring buffer.
2. The MAC raises a hardware interrupt to the Generic Interrupt Controller (GIC).
3. The GIC forwards the interrupt to the ARM CPU.
4. The CPU executes the driver's interrupt handler (`emacps_recv_handler`), which takes the descriptor from the DMA ring and pushes it into a software queue (`recv_q`).
5. The `while(1)` loop calls `xemacif_input()`, which pulls packets from `recv_q` and processes them (replying to ARP, ICMP Ping, etc.).

### The Failure Point
The application initialized lwIP and successfully connected the MAC interrupt to the GIC using `XSetupInterruptSystem()`. However, the code only called `Xil_ExceptionEnable()` to enable interrupts in the ARM CPU's CPSR register. 

Crucially, it **failed to register the GIC's master exception handler to the ARM CPU**. 

Because the master handler (`XScuGic_DeviceInterruptHandler`) was never registered, the ARM CPU silently dropped all incoming hardware interrupts. As a result:
* The MAC received the ARP requests and raised an interrupt.
* The CPU ignored the interrupt.
* `emacps_recv_handler` never ran.
* `recv_q` remained empty.
* `xemacif_input()` saw an empty queue and did nothing.

## 4. The Fix

By comparing the codebase to a verified working infrastructure (`delhatch/Zynq_UDP`), we identified the missing platform-level initialization. 

The fix involved modifying `main.c` to correctly initialize the interrupt ecosystem:

1. **Platform Interrupts:** Replaced the low-level `Xil_ExceptionEnable()` with `platform_enable_interrupts()`. This function natively handles registering the GIC Master Handler to the ARM exception vector and starting the SCU Timer.
2. **lwIP Timers:** Added the SCU Timer flags (`TcpFastTmrFlag` and `TcpSlowTmrFlag`) and `sys_check_timeouts()` to the `while(1)` loop. This ensures that lwIP's periodic background tasks (such as managing the ARP cache) are executed correctly.

With the interrupts fully bridged to the CPU, the RX handler can populate the software queues, allowing `xemacif_input()` to read incoming packets and respond to network traffic properly.
