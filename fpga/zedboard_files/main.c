#include <stdio.h>
#include "xparameters.h"
#include "netif/xadapter.h"
#include "platform.h"
#include "platform_config.h"
#include "lwip/udp.h"
#include "xaxidma.h"

// --- Configuration ---
#define DMA_DEV_ID          XPAR_XAXIVDMA_0_BASEADDR
#define RX_BUFFER_BASEADDR  0x01000000 // A safe spot in DDR memory
#define ROW_BYTES           1280       // 640 pixels * 2 bytes
#define UDP_TARGET_PORT     5005

// Replace with the Static IP of your PC's Ethernet port
#define PC_IP_ADDR          "192.168.1.100" 
#define ZYNQ_IP_ADDR        "192.168.1.10"
#define NETMASK             "255.255.255.0"
#define GATEWAY             "192.168.1.1"

// Global Instances
XAxiDma AxiDma;
struct netif server_netif;
struct udp_pcb *vid_pcb;
ip_addr_t target_ip;

void print_ip(char *msg, ip_addr_t *ip) {
    xil_printf(msg);
    xil_printf("%d.%d.%d.%d\n\r", ip4_addr1(ip), ip4_addr2(ip), ip4_addr3(ip), ip4_addr4(ip));
}

int init_dma() {
    XAxiDma_Config *CfgPtr;
    int Status;

    CfgPtr = XAxiDma_LookupConfig(DMA_DEV_ID);
    if (!CfgPtr) {
        xil_printf("No config found for DMA\n\r");
        return XST_FAILURE;
    }

    Status = XAxiDma_CfgInitialize(&AxiDma, CfgPtr);
    if (Status != XST_SUCCESS) {
        xil_printf("Initialization failed\n\r");
        return XST_FAILURE;
    }

    if(XAxiDma_HasSg(&AxiDma)){
        xil_printf("Device configured as SG mode, this script requires Simple mode\n\r");
        return XST_FAILURE;
    }

    // Disable interrupts, we will use polling for this test
    XAxiDma_IntrDisable(&AxiDma, XAXIDMA_IRQ_ALL_MASK, XAXIDMA_DEVICE_TO_DMA);
    return XST_SUCCESS;
}

int main() {
    ip_addr_t ipaddr, netmask, gw;
    struct pbuf *p;
    int Status;

    init_platform();

    xil_printf("\n\r--- ZedBoard OV7670 UDP Streamer ---\n\r");

    // 1. Initialize DMA
    if (init_dma() != XST_SUCCESS) {
        return -1;
    }
    xil_printf("AXI DMA Initialized.\n\r");

    // 2. Initialize Network (lwIP)
    ipaddr.addr  = inet_addr(ZYNQ_IP_ADDR);
    netmask.addr = inet_addr(NETMASK);
    gw.addr      = inet_addr(GATEWAY);
    target_ip.addr = inet_addr(PC_IP_ADDR);

    lwip_init();

    // Add network interface to the netif_list, and set it as default
    if (!xemac_add(&server_netif, &ipaddr, &netmask, &gw, (unsigned char[]){0x00, 0x0a, 0x35, 0x00, 0x01, 0x02}, XPAR_XEMACPS_0_BASEADDR)) {
        xil_printf("Error adding N/W interface\n\r");
        return -1;
    }
    netif_set_default(&server_netif);
    netif_set_up(&server_netif);
    
    print_ip("Board IP: ", &ipaddr);
    print_ip("Target PC IP: ", &target_ip);

    // 3. Set up UDP Protocol Control Block (PCB)
    vid_pcb = udp_new();
    if (!vid_pcb) {
        xil_printf("Error creating PCB\n\r");
        return -1;
    }

    u8 *RxBufferPtr = (u8 *)RX_BUFFER_BASEADDR;

    xil_printf("Starting Video Stream...\n\r");

    // 4. Main Streaming Loop
    while (1) {
        // Must call this frequently to keep lwIP stack ticking
        xemacif_input(&server_netif);

        // Step A: Trigger DMA to pull 1 row (1280 bytes) from PL to DDR
        Status = XAxiDma_SimpleTransfer(&AxiDma, (u32)RxBufferPtr, ROW_BYTES, XAXIDMA_DEVICE_TO_DMA);
        if (Status != XST_SUCCESS) {
            continue; // Handle error or retry
        }

        // Step B: Wait for DMA transfer to finish (Polling)
        while (XAxiDma_Busy(&AxiDma, XAXIDMA_DEVICE_TO_DMA)) {
            // Keep network alive while waiting
            xemacif_input(&server_netif); 
        }

        // Step C: Flush CPU Cache so we see the fresh hardware data
        Xil_DCacheInvalidateRange((UINTPTR)RxBufferPtr, ROW_BYTES);

        // Step D: Allocate lwIP packet buffer and copy data
        p = pbuf_alloc(PBUF_TRANSPORT, ROW_BYTES, PBUF_POOL);
        if (p != NULL) {
            memcpy(p->payload, RxBufferPtr, ROW_BYTES);
            
            // Send the packet blind to the PC
            udp_sendto(vid_pcb, p, &target_ip, UDP_TARGET_PORT);
            
            pbuf_free(p);
        }
    }

    cleanup_platform();
    return 0;
}