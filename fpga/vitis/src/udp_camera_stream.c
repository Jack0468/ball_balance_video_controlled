#include <stdio.h>
#include <string.h>
#include "xparameters.h"
#include "xaxivdma.h"
#include "xil_cache.h"
#include "lwip/udp.h"
#include "lwip/init.h"
#include "netif/xadapter.h"
// Removed platform.h inclusions

// Hardware settings
#define VDMA_ID          XPAR_AXI_VDMA_0_DEVICE_ID
#define FRAME_BUFFER_ADDR 0x10000000 // DDR3 Address for Video Frame
#define WIDTH 640
#define HEIGHT 480
#define PIXEL_BYTES 2
#define FRAME_SIZE (WIDTH * HEIGHT * PIXEL_BYTES)

// Network settings
#define PLATFORM_EMAC_BASEADDR XPAR_XEMACPS_0_BASEADDR

// MAKE SURE THIS MATCHES YOUR PC'S IP ADDRESS ON THE ETHERNET ADAPTER!
#define PC_IP_1 192
#define PC_IP_2 168
#define PC_IP_3 1
#define PC_IP_4 100
#define PC_UDP_PORT 8080
#define PACKET_PAYLOAD_SIZE 1024

XAxiVdma vdma;
struct netif server_netif;

// Simple Header for UDP Packets
struct PacketHeader {
    uint32_t frame_id;
    uint32_t packet_id;
};

void init_vdma() {
    XAxiVdma_Config *config = XAxiVdma_LookupConfig(VDMA_ID);
    if (!config) {
        xil_printf("No VDMA found!\r\n");
        return;
    }
    XAxiVdma_CfgInitialize(&vdma, config, config->BaseAddress);

    XAxiVdma_DmaSetup setup;
    setup.VertSizeInput = HEIGHT;
    setup.HoriSizeInput = WIDTH * PIXEL_BYTES;
    setup.Stride = WIDTH * PIXEL_BYTES;
    setup.FrameDelay = 0;
    setup.EnableCircularBuf = 1; // Continuous capture
    setup.EnableSync = 0;
    setup.PointNum = 0;
    setup.EnableFrameCounter = 0;
    setup.FixedFrameStoreAddr = 0; // Use index 0

    XAxiVdma_DmaConfig(&vdma, XAXIVDMA_WRITE, &setup);

    UINTPTR addr = FRAME_BUFFER_ADDR;
    XAxiVdma_DmaSetBufferAddr(&vdma, XAXIVDMA_WRITE, &addr);

    XAxiVdma_DmaStart(&vdma, XAXIVDMA_WRITE);
    xil_printf("VDMA Started!\r\n");
}

void send_frame_udp(struct udp_pcb *pcb, ip_addr_t *dest_ip, uint32_t frame_id) {
    uint8_t *frame_ptr = (uint8_t *)FRAME_BUFFER_ADDR;
    uint32_t bytes_sent = 0;
    uint32_t packet_id = 0;
    
    // Critical: Invalidate Data Cache so the ARM sees the latest pixels written by the VDMA!
    Xil_DCacheInvalidateRange(FRAME_BUFFER_ADDR, FRAME_SIZE);

    while (bytes_sent < FRAME_SIZE) {
        uint32_t to_send = FRAME_SIZE - bytes_sent;
        if (to_send > PACKET_PAYLOAD_SIZE) {
            to_send = PACKET_PAYLOAD_SIZE;
        }

        // Allocate pbuf for UDP
        struct pbuf *p = pbuf_alloc(PBUF_TRANSPORT, sizeof(struct PacketHeader) + to_send, PBUF_RAM);
        if (p != NULL) {
            struct PacketHeader *hdr = (struct PacketHeader *)p->payload;
            hdr->frame_id = frame_id;
            hdr->packet_id = packet_id;
            
            // Copy pixels into packet
            memcpy((uint8_t *)p->payload + sizeof(struct PacketHeader), frame_ptr + bytes_sent, to_send);
            
            // Send to PC
            udp_sendto(pcb, p, dest_ip, PC_UDP_PORT);
            pbuf_free(p);
        }

        bytes_sent += to_send;
        packet_id++;
    }
}

int main() {
    Xil_ICacheEnable();
    Xil_DCacheEnable();
    xil_printf("ZedBoard OV7670 Video Streamer\r\n");
    
    // Init VDMA
    init_vdma();
    
    // Init Network (lwIP)
    lwip_init();
    
    ip_addr_t ipaddr, netmask, gw;
    // ZedBoard IP
    IP4_ADDR(&ipaddr, 192, 168, 1, 10);
    IP4_ADDR(&netmask, 255, 255, 255, 0);
    IP4_ADDR(&gw, 192, 168, 1, 1);
    
    // Mac Address (Random local MAC)
    unsigned char mac_ethernet_address[] = { 0x00, 0x0a, 0x35, 0x00, 0x01, 0x02 };
    
    xemac_add(&server_netif, &ipaddr, &netmask, &gw, mac_ethernet_address, PLATFORM_EMAC_BASEADDR);
    netif_set_default(&server_netif);
    netif_set_up(&server_netif);
    xil_printf("Network Ready. IP: 192.168.1.10\r\n");
    
    struct udp_pcb *pcb = udp_new();
    ip_addr_t pc_ip;
    IP4_ADDR(&pc_ip, PC_IP_1, PC_IP_2, PC_IP_3, PC_IP_4);
    
    uint32_t frame_id = 0;
    xil_printf("Starting UDP Video Stream to PC...\r\n");
    
    while (1) {
        // Send a frame
        send_frame_udp(pcb, &pc_ip, frame_id);
        frame_id++;
        
        // Call lwIP periodic tasks (if required by the port, though usually in a timer interrupt)
        // Delay to hit ~30 FPS. At 667MHz, 10 million loop iterations is roughly a few dozen ms.
        for(volatile int i=0; i<5000000; i++); 
    }
    
    Xil_DCacheDisable();
    Xil_ICacheDisable();
    return 0;
}
