`timescale 1ns / 1ps
//=============================================================================
// optical_bridge_axi_lite.v
//
// *** PRELIMINARY / SPECULATIVE — DO NOT TREAT AS A REAL DESIGN ***
//
// This module is a BEHAVIORAL SKETCH of what an AXI4-Lite-addressable
// digital<->optical bridge peripheral MIGHT look like for Arm 3 of this
// project's 3-arm VLA comparison, IF framing (a) in
// fpga/docs/optical_bridge_interface_spec_draft.md section 1 turns out to be
// true (a real fabricated photonic chip this project has electrical access
// to). That framing question is STILL UNRESOLVED as of 2026-10-14 — see the
// spec draft and this directory's README before reading anything below as a
// committed design.
//
// This file exists so that when a real spec does land, there is a concrete
// skeleton to edit rather than a blank page — NOT because any part of the
// shape below has been decided. Every "placeholder" parameter/constant is
// named and commented at the point it is defined; do not let any of them
// quietly become load-bearing in a later design.
//
// Two things this module deliberately does NOT attempt (see README for the
// full list):
//   - No real DAC/ADC bus protocol. The dac_bus_*/adc_bus_* ports below are a
//     generic placeholder parallel bus invented for this sketch only — not
//     SPI, not I2C, not any real part's interface. No DAC/ADC part has been
//     chosen (spec draft S4, UNKNOWN).
//   - No real settling-time modeling. SETTLE_CYCLES_PLACEHOLDER below is a
//     made-up simulation-only wait count, NOT a real thermo-optic/EO
//     actuator settling time — the spec draft S3.3 explicitly flags that
//     number as a genuine unknown, not something to guess from photonics
//     literature recall.
//
// Interface-packaging precedent: this follows the same AXI4-Lite
// register-wrapper shape already used/in-progress for
// fpga/camera_i2c/ov7670_axi_stream.v (packaged into zynq_camera_sys as a
// custom IP, AXI4-Stream though, not AXI4-Lite) and the in-progress Domain 4
// AXI4-Lite port of fpga/main_controller/stepper_motor_controller.v. That
// precedent is why this sketch is shaped as an AXI4-Lite slave + internal
// register file + FSM, rather than any protocol-level claim about the real
// photonic bridge.
//=============================================================================

module optical_bridge_axi_lite #(
    // -------------------------------------------------------------------
    // PLACEHOLDER PARAMETERS — every one of these is a stand-in, not a
    // decision. Defaults below are the source paper's *example* fabricated
    // device sizes (Sved et al., MNIST device: N=10, C=10) per
    // optical_bridge_interface_spec_draft.md S3.1/S7, chosen ONLY because
    // they are the only concrete numbers available anywhere in this
    // project's research material — NOT because any project device of this
    // size exists or has been chosen. See spec draft S8 Q2.
    // -------------------------------------------------------------------
    parameter integer N                = 10,  // PLACEHOLDER: input channel count (paper example, UNKNOWN for a real project device)
    parameter integer C                = 10,  // PLACEHOLDER: output/readback channel count (paper example, UNKNOWN for a real project device)
    parameter integer DAC_WIDTH         = 12,  // PLACEHOLDER: spec draft S3.2 DERIVED ~11 bits from the paper's calibration sweep granularity; rounded up, not a real DAC spec
    parameter integer ADC_WIDTH         = 12,  // PLACEHOLDER: no ADC/photodetector part chosen anywhere (spec draft S4, UNKNOWN) — arbitrary same-width guess for symmetry only
    parameter integer AXI_ADDR_WIDTH    = 16,  // PLACEHOLDER: wide enough for this sketch's register map (S5 below), not sized against a real integration
    parameter integer SETTLE_CYCLES_PLACEHOLDER = 16, // PLACEHOLDER: simulation-only wait, NOT a real actuator settling time (spec draft S3.3, UNKNOWN)
    parameter integer ADC_LATENCY_CYCLES_PLACEHOLDER = 4 // PLACEHOLDER: simulation-only wait between asserting an ADC read address and sampling data back
) (
    input  wire                        s_axi_aclk,
    input  wire                        s_axi_aresetn,

    // AXI4-Lite write address channel
    input  wire [AXI_ADDR_WIDTH-1:0]   s_axi_awaddr,
    input  wire                        s_axi_awvalid,
    output wire                        s_axi_awready,

    // AXI4-Lite write data channel
    input  wire [31:0]                 s_axi_wdata,
    input  wire [3:0]                  s_axi_wstrb,   // NOTE: simplification — full-word writes only, see README
    input  wire                        s_axi_wvalid,
    output wire                        s_axi_wready,

    // AXI4-Lite write response channel
    output wire [1:0]                  s_axi_bresp,
    output wire                        s_axi_bvalid,
    input  wire                        s_axi_bready,

    // AXI4-Lite read address channel
    input  wire [AXI_ADDR_WIDTH-1:0]   s_axi_araddr,
    input  wire                        s_axi_arvalid,
    output wire                        s_axi_arready,

    // AXI4-Lite read data channel
    output wire [31:0]                 s_axi_rdata,
    output wire [1:0]                  s_axi_rresp,
    output wire                        s_axi_rvalid,
    input  wire                        s_axi_rready,

    // -------------------------------------------------------------------
    // PLACEHOLDER physical-side bus — invented for this sketch only.
    // NOT a real DAC/ADC protocol. A real spec will very likely replace
    // this entirely (SPI, parallel, vendor-specific, whatever the chosen
    // part needs) — see README "what changes once a real part is known".
    // -------------------------------------------------------------------
    output wire [$clog2(2*N-1)-1:0]    dac_bus_addr,
    output wire [DAC_WIDTH-1:0]        dac_bus_data,
    output wire                        dac_bus_wr_en,

    output wire [$clog2(C)-1:0]        adc_bus_addr,
    output wire                        adc_bus_rd_en,
    input  wire [ADC_WIDTH-1:0]        adc_bus_data_in,

    // Status, exposed for the testbench / future PS-side polling convenience
    output wire                        busy,
    output wire                        done
);

    // =====================================================================
    // Derived sizing (not independently a placeholder; purely arithmetic on
    // the placeholders above)
    // =====================================================================
    localparam integer NUM_DAC_ENTRIES = 2*N - 1;      // CITED relationship (spec draft S3.1): N amplitude + (N-1) phase channels
    localparam integer NUM_ADC_ENTRIES = C;
    localparam integer DAC_IDX_WIDTH   = (NUM_DAC_ENTRIES <= 1) ? 1 : $clog2(NUM_DAC_ENTRIES);
    localparam integer ADC_IDX_WIDTH   = (NUM_ADC_ENTRIES <= 1) ? 1 : $clog2(NUM_ADC_ENTRIES);

    // =====================================================================
    // PROPOSED register map — arbitrary word layout invented for this
    // sketch, not reviewed/decided by anyone. Word-addressed (bits [1:0] of
    // the AXI address are ignored, i.e. this only supports aligned 32-bit
    // accesses).
    //
    //   0x0000  CTRL    [0]=START (self-clearing write-1-to-pulse)
    //   0x0004  STATUS  [0]=BUSY  [1]=DONE  [15:8]=current FSM sequence index (debug)
    //   0x0008  VERSION read-only sanity word (0xCAFEBEEF), lets a host confirm
    //           it is actually talking to this peripheral before trusting the rest
    //   0x0100.. DAC_TABLE   : NUM_DAC_ENTRIES words, one DAC value per word
    //   0x0400.. ADC_TABLE   : NUM_ADC_ENTRIES words, one ADC readback value per word (read-only)
    // =====================================================================
    localparam [AXI_ADDR_WIDTH-1:0] ADDR_CTRL     = 16'h0000;
    localparam [AXI_ADDR_WIDTH-1:0] ADDR_STATUS   = 16'h0004;
    localparam [AXI_ADDR_WIDTH-1:0] ADDR_VERSION  = 16'h0008;
    localparam [AXI_ADDR_WIDTH-1:0] ADDR_DAC_BASE = 16'h0100;
    localparam [AXI_ADDR_WIDTH-1:0] ADDR_ADC_BASE = 16'h0400;
    localparam [31:0]               VERSION_MAGIC = 32'hCAFEBEEF;

    // =====================================================================
    // AXI4-Lite slave protocol handling — deliberately minimal single-beat
    // handshaking (no pipelining, no outstanding transactions). Sufficient
    // for a register file this small; not claimed to be a general-purpose
    // AXI4-Lite slave implementation.
    // =====================================================================
    reg                        awready_r, wready_r, bvalid_r;
    reg [AXI_ADDR_WIDTH-1:0]   awaddr_latched;
    reg                        arready_r, rvalid_r;
    reg [31:0]                 rdata_r;

    assign s_axi_awready = awready_r;
    assign s_axi_wready  = wready_r;
    assign s_axi_bresp   = 2'b00; // OKAY, always
    assign s_axi_bvalid  = bvalid_r;
    assign s_axi_arready = arready_r;
    assign s_axi_rdata   = rdata_r;
    assign s_axi_rresp   = 2'b00; // OKAY, always
    assign s_axi_rvalid  = rvalid_r;

    // Internal register file
    reg [DAC_WIDTH-1:0] dac_table [0:NUM_DAC_ENTRIES-1];
    reg [ADC_WIDTH-1:0] adc_table [0:NUM_ADC_ENTRIES-1];

    reg        ctrl_start_pulse;
    reg        busy_r, done_r;
    reg [7:0]  seq_idx_dbg;

    assign busy = busy_r;
    assign done = done_r;

    integer di; // for reset loop only (simulation convenience)
    // ---------------------------------------------------------------
    // Write channel FSM
    // ---------------------------------------------------------------
    reg write_active;
    wire [AXI_ADDR_WIDTH-1:0] waddr_word = awaddr_latched; // word-aligned assumption (S/w must write aligned addrs)

    always @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            awready_r        <= 1'b1;
            wready_r         <= 1'b0;
            bvalid_r         <= 1'b0;
            write_active     <= 1'b0;
            ctrl_start_pulse <= 1'b0;
            for (di = 0; di < NUM_DAC_ENTRIES; di = di + 1)
                dac_table[di] <= {DAC_WIDTH{1'b0}};
        end else begin
            ctrl_start_pulse <= 1'b0; // default: self-clearing, one cycle pulse

            // Accept write address
            if (s_axi_awvalid && awready_r) begin
                awaddr_latched <= s_axi_awaddr;
                awready_r      <= 1'b0;
                wready_r       <= 1'b1;
                write_active   <= 1'b1;
            end

            // Accept write data once address latched
            if (write_active && s_axi_wvalid && wready_r) begin
                wready_r <= 1'b0;
                bvalid_r <= 1'b1;

                if (waddr_word == ADDR_CTRL) begin
                    if (s_axi_wdata[0])
                        ctrl_start_pulse <= 1'b1;
                end else if (waddr_word >= ADDR_DAC_BASE &&
                             waddr_word < (ADDR_DAC_BASE + 4*NUM_DAC_ENTRIES)) begin
                    dac_table[(waddr_word - ADDR_DAC_BASE) >> 2] <= s_axi_wdata[DAC_WIDTH-1:0];
                end
                // Writes to STATUS/VERSION/ADC_TABLE (read-only regions) are
                // accepted at the AXI protocol level (still ack'd, OKAY) but
                // silently dropped — no real register backs them for write.
            end

            // Write response handshake
            if (bvalid_r && s_axi_bready) begin
                bvalid_r     <= 1'b0;
                write_active <= 1'b0;
                awready_r    <= 1'b1;
            end
        end
    end

    // ---------------------------------------------------------------
    // Read channel FSM
    // ---------------------------------------------------------------
    always @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            arready_r <= 1'b1;
            rvalid_r  <= 1'b0;
            rdata_r   <= 32'h0;
        end else begin
            if (s_axi_arvalid && arready_r) begin
                arready_r <= 1'b0;
                rvalid_r  <= 1'b1;

                if (s_axi_araddr == ADDR_CTRL) begin
                    rdata_r <= 32'h0; // CTRL reads back as 0 (write-only pulse register)
                end else if (s_axi_araddr == ADDR_STATUS) begin
                    rdata_r <= {16'h0, seq_idx_dbg, 6'h0, done_r, busy_r};
                end else if (s_axi_araddr == ADDR_VERSION) begin
                    rdata_r <= VERSION_MAGIC;
                end else if (s_axi_araddr >= ADDR_DAC_BASE &&
                             s_axi_araddr < (ADDR_DAC_BASE + 4*NUM_DAC_ENTRIES)) begin
                    rdata_r <= {{(32-DAC_WIDTH){1'b0}}, dac_table[(s_axi_araddr - ADDR_DAC_BASE) >> 2]};
                end else if (s_axi_araddr >= ADDR_ADC_BASE &&
                             s_axi_araddr < (ADDR_ADC_BASE + 4*NUM_ADC_ENTRIES)) begin
                    rdata_r <= {{(32-ADC_WIDTH){1'b0}}, adc_table[(s_axi_araddr - ADDR_ADC_BASE) >> 2]};
                end else begin
                    rdata_r <= 32'hDEAD_DEAD; // unmapped address sentinel
                end
            end else if (rvalid_r && s_axi_rready) begin
                rvalid_r  <= 1'b0;
                arready_r <= 1'b1;
            end
        end
    end

    // =====================================================================
    // Write-then-read sequencing FSM
    //
    // Behavioral shape only: on START, walk every DAC table entry out over
    // the placeholder dac_bus (one entry per cycle — real per-channel write
    // timing is UNKNOWN, see spec draft S3.3), wait a placeholder settle
    // period, then walk every ADC table entry in over the placeholder
    // adc_bus. This is the "simple write-then-read sequencing FSM" shape
    // requested — it is NOT a claim about real per-channel timing,
    // parallelism (a real bridge might write/read channels in parallel,
    // not serially), or bus protocol.
    // =====================================================================
    localparam [2:0] S_IDLE        = 3'd0,
                      S_WRITE_DAC   = 3'd1,
                      S_SETTLE      = 3'd2,
                      S_READ_ADC    = 3'd3,
                      S_ADC_LATCH   = 3'd4,
                      S_DONE        = 3'd5;

    reg [2:0]                  state;
    reg [DAC_IDX_WIDTH-1:0]    dac_idx;
    reg [ADC_IDX_WIDTH-1:0]    adc_idx;
    reg [31:0]                 wait_ctr;

    reg [DAC_IDX_WIDTH-1:0]    dac_bus_addr_r;
    reg [DAC_WIDTH-1:0]        dac_bus_data_r;
    reg                        dac_bus_wr_en_r;
    reg [ADC_IDX_WIDTH-1:0]    adc_bus_addr_r;
    reg                        adc_bus_rd_en_r;

    assign dac_bus_addr  = dac_bus_addr_r;
    assign dac_bus_data  = dac_bus_data_r;
    assign dac_bus_wr_en = dac_bus_wr_en_r;
    assign adc_bus_addr  = adc_bus_addr_r;
    assign adc_bus_rd_en = adc_bus_rd_en_r;

    integer ai; // reset loop only
    always @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            state           <= S_IDLE;
            dac_idx         <= {DAC_IDX_WIDTH{1'b0}};
            adc_idx         <= {ADC_IDX_WIDTH{1'b0}};
            wait_ctr        <= 32'd0;
            busy_r          <= 1'b0;
            done_r          <= 1'b0;
            seq_idx_dbg     <= 8'd0;
            dac_bus_wr_en_r <= 1'b0;
            adc_bus_rd_en_r <= 1'b0;
            dac_bus_addr_r  <= {DAC_IDX_WIDTH{1'b0}};
            dac_bus_data_r  <= {DAC_WIDTH{1'b0}};
            adc_bus_addr_r  <= {ADC_IDX_WIDTH{1'b0}};
            for (ai = 0; ai < NUM_ADC_ENTRIES; ai = ai + 1)
                adc_table[ai] <= {ADC_WIDTH{1'b0}};
        end else begin
            dac_bus_wr_en_r <= 1'b0;
            adc_bus_rd_en_r <= 1'b0;

            case (state)
                S_IDLE: begin
                    busy_r <= 1'b0;
                    if (ctrl_start_pulse) begin
                        busy_r  <= 1'b1;
                        done_r  <= 1'b0;
                        dac_idx <= {DAC_IDX_WIDTH{1'b0}};
                        state   <= S_WRITE_DAC;
                    end
                end

                S_WRITE_DAC: begin
                    seq_idx_dbg     <= {5'b0, S_WRITE_DAC};
                    dac_bus_addr_r  <= dac_idx;
                    dac_bus_data_r  <= dac_table[dac_idx];
                    dac_bus_wr_en_r <= 1'b1;
                    if (dac_idx == NUM_DAC_ENTRIES-1) begin
                        wait_ctr <= 32'd0;
                        state    <= S_SETTLE;
                    end else begin
                        dac_idx <= dac_idx + 1'b1;
                    end
                end

                S_SETTLE: begin
                    seq_idx_dbg <= {5'b0, S_SETTLE};
                    if (wait_ctr == SETTLE_CYCLES_PLACEHOLDER - 1) begin
                        adc_idx  <= {ADC_IDX_WIDTH{1'b0}};
                        state    <= S_READ_ADC;
                    end else begin
                        wait_ctr <= wait_ctr + 1'b1;
                    end
                end

                S_READ_ADC: begin
                    seq_idx_dbg    <= {5'b0, S_READ_ADC};
                    adc_bus_addr_r <= adc_idx;
                    adc_bus_rd_en_r<= 1'b1;
                    wait_ctr       <= 32'd0;
                    state          <= S_ADC_LATCH;
                end

                S_ADC_LATCH: begin
                    seq_idx_dbg <= {5'b0, S_ADC_LATCH};
                    if (wait_ctr == ADC_LATENCY_CYCLES_PLACEHOLDER - 1) begin
                        adc_table[adc_idx] <= adc_bus_data_in;
                        if (adc_idx == NUM_ADC_ENTRIES-1) begin
                            state <= S_DONE;
                        end else begin
                            adc_idx <= adc_idx + 1'b1;
                            state   <= S_READ_ADC;
                        end
                    end else begin
                        wait_ctr <= wait_ctr + 1'b1;
                    end
                end

                S_DONE: begin
                    seq_idx_dbg <= {5'b0, S_DONE};
                    busy_r      <= 1'b0;
                    done_r      <= 1'b1;
                    state       <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
