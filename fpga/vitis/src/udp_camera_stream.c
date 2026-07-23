#include <stdio.h>
#include <string.h>
#include "xparameters.h"
#include "xaxivdma.h"
#include "xil_cache.h"
#include "lwip/udp.h"
#include "lwip/init.h"
#include "netif/xadapter.h"
#include "platform.h"
#include "platform_config.h"

extern volatile int TcpFastTmrFlag;
extern volatile int TcpSlowTmrFlag;

void tcp_fasttmr(void);
void tcp_slowtmr(void);

// Hardware settings
#define VDMA_ID          XPAR_XAXIVDMA_0_BASEADDR
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
#define PC_UDP_PORT 5001
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
    setup.HoriSizeInput = WIDTH * PIXEL_BYTES;
    setup.VertSizeInput = HEIGHT;
    setup.Stride = WIDTH * PIXEL_BYTES;
    setup.FrameDelay = 0;
    setup.EnableCircularBuf = 1; // Continuous capture
    setup.EnableSync = 0;
    setup.PointNum = 0;
    setup.EnableFrameCounter = 1;
    setup.FixedFrameStoreAddr = 0; // Use index 0

    XAxiVdma_DmaConfig(&vdma, XAXIVDMA_WRITE, &setup);

    UINTPTR addr[3] = {
        FRAME_BUFFER_ADDR, 
        FRAME_BUFFER_ADDR + FRAME_SIZE, 
        FRAME_BUFFER_ADDR + 2 * FRAME_SIZE
    };
    XAxiVdma_DmaSetBufferAddr(&vdma, XAXIVDMA_WRITE, addr);

    XAxiVdma_DmaStart(&vdma, XAXIVDMA_WRITE);
    
    // Enable frame count interrupt bit (so we can poll the status register)
    XAxiVdma_IntrEnable(&vdma, XAXIVDMA_IXR_FRMCNT_MASK, XAXIVDMA_WRITE);
    // Tell VDMA to trigger the interrupt every 1 frame
    XAxiVdma_FrameCounter FrameCfg;
    FrameCfg.ReadFrameCount = 1;
    FrameCfg.WriteFrameCount = 1;
    FrameCfg.ReadDelayTimerCount = 0;
    FrameCfg.WriteDelayTimerCount = 0;
    XAxiVdma_SetFrameCounter(&vdma, &FrameCfg);
    
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
    init_platform();
    xil_printf("ZedBoard OV7670 Video Streamer\r\n");
    
    // Init Network (lwIP) first! If VDMA freezes, network will at least ping!
    lwip_init();
    
    ip_addr_t ipaddr, netmask, gw;
    // ZedBoard IP
    IP4_ADDR(&ipaddr, 192, 168, 1, 10);
    IP4_ADDR(&netmask, 255, 255, 255, 0);
    IP4_ADDR(&gw, 192, 168, 1, 1);
    
    // Mac Address (Random local MAC)
    unsigned char mac_ethernet_address[] = { 0x00, 0x0a, 0x35, 0x00, 0x01, 0x02 };
    
    if (!xemac_add(&server_netif, &ipaddr, &netmask, &gw, mac_ethernet_address, PLATFORM_EMAC_BASEADDR)) {
        xil_printf("MAC INIT FAILED\r\n");
    }
    netif_set_default(&server_netif);
    netif_set_up(&server_netif);
    
    // FORCING LINK UP: If eth_link_detect is removed, lwIP might drop all outgoing packets
    // thinking the cable is disconnected! This forces lwIP to respond to ARP.
    netif_set_link_up(&server_netif);
    
    // Explicitly enable interrupts on ARM (Required in SDT after all handlers are registered)
    #include "xil_exception.h"
    Xil_ExceptionEnable();
    
    xil_printf("Network Ready. IP: 192.168.1.10\r\n");
    
    // Now init VDMA
    init_vdma();
    
    struct udp_pcb *pcb = udp_new();
    ip_addr_t pc_ip;
    IP4_ADDR(&pc_ip, PC_IP_1, PC_IP_2, PC_IP_3, PC_IP_4);
    
    uint32_t frame_id = 0;
    uint32_t delay_count = 0;
    xil_printf("Starting UDP Video Stream to PC...\r\n");
    
    while (1) {
        // Check timers for lwIP
        if (TcpFastTmrFlag) {
            tcp_fasttmr();
            TcpFastTmrFlag = 0;
        }
        if (TcpSlowTmrFlag) {
            tcp_slowtmr();
            TcpSlowTmrFlag = 0;
        }
        
        // CRITICAL: We must constantly poll the Ethernet RX buffer to process incoming ARP and PING packets!
        xemacif_input(&server_netif);
        
        // Poll VDMA status to see if a frame finished
        u32 vdma_status = XAxiVdma_IntrGetPending(&vdma, XAXIVDMA_WRITE);
        
        if (delay_count++ > 5000000) {
            u32 raw_status = XAxiVdma_GetStatus(&vdma, XAXIVDMA_WRITE);
            xil_printf("DEBUG: VDMA Pending Intr: %08x, Raw Status: %08x\r\n", vdma_status, raw_status);
            delay_count = 0;
        }
        
        if (vdma_status & XAXIVDMA_IXR_FRMCNT_MASK) {
            // Clear the interrupt status bit so we can catch the next frame
            XAxiVdma_IntrClear(&vdma, XAXIVDMA_IXR_FRMCNT_MASK, XAXIVDMA_WRITE);
            
            // Send the freshly captured frame perfectly synced to 30 FPS!
            send_frame_udp(pcb, &pc_ip, frame_id);
            frame_id++;
        }
    }
    
    cleanup_platform();
    return 0;
}
