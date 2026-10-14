`timescale 1ns / 1ps
//=============================================================================
// tb_optical_bridge_axi_lite.v
//
// SIMULATION-ONLY testbench for optical_bridge_axi_lite.v.
//
// *** This exercises the PRELIMINARY/SPECULATIVE sketch module only. It does
// not touch, model, or validate any real DAC/ADC/photonic hardware — there
// is none to model yet. See ../README.md and
// fpga/docs/optical_bridge_interface_spec_draft.md before treating a PASS
// here as validating anything beyond "this behavioral sketch's own internal
// FSM/register logic is self-consistent." ***
//
// What this testbench actually checks, per the task's requirement to
// "confirm the FSM sequencing and register read/write behavior in
// isolation":
//   1. VERSION register sanity readback.
//   2. Every DAC_TABLE word written over AXI4-Lite reads back unchanged.
//   3. Triggering START walks the placeholder dac_bus through every index
//      0..NUM_DAC_ENTRIES-1 exactly once, in order (monitored directly on
//      the dac_bus_* ports, independent of the AXI side).
//   4. After the settle wait, the placeholder adc_bus is walked through
//      every index 0..NUM_ADC_ENTRIES-1 exactly once, in order, and the
//      value presented on adc_bus_data_in at each address is correctly
//      latched into ADC_TABLE (confirmed by reading ADC_TABLE back over
//      AXI4-Lite afterward).
//   5. BUSY/DONE status bits behave as expected across the sequence.
//   6. The whole sequence can be re-triggered a second time without a
//      reset, to catch any state that fails to rearm.
//
// Uses small, non-default parameters (N=3, C=2) so the DAC/ADC sequence
// lengths are short enough to read the simulation waveform/log by hand, then
// repeats the same checks once more with the module's PLACEHOLDER default
// sizes (N=10, C=10) to confirm the FSM also behaves at the size the module
// defaults to.
//=============================================================================

module tb_optical_bridge_axi_lite;

    // -------------------------------------------------------------------
    // Clock / reset
    // -------------------------------------------------------------------
    reg clk = 1'b0;
    reg aresetn = 1'b0;
    always #5 clk = ~clk; // 100MHz placeholder sim clock -- NOT tied to any real board clock plan

    // -------------------------------------------------------------------
    // Shared error counter
    // -------------------------------------------------------------------
    integer errors = 0;

    // =====================================================================
    // Instance 1: small sizes (N=3, C=2) for an easy-to-read trace
    // =====================================================================
    localparam N0 = 3, C0 = 2;
    localparam AXI_AW0 = 16;
    localparam DACW0 = 12, ADCW0 = 12;

    reg  [AXI_AW0-1:0] awaddr0; reg awvalid0; wire awready0;
    reg  [31:0]        wdata0;  reg wvalid0;  wire wready0;
    wire [1:0]          bresp0; wire bvalid0; reg bready0;
    reg  [AXI_AW0-1:0] araddr0; reg arvalid0; wire arready0;
    wire [31:0]         rdata0; wire [1:0] rresp0; wire rvalid0; reg rready0;

    wire [$clog2(2*N0-1)-1:0] dac_addr0;
    wire [DACW0-1:0]           dac_data0;
    wire                        dac_wr0;
    wire [$clog2(C0)-1:0]      adc_addr0;
    wire                        adc_rd0;
    reg  [ADCW0-1:0]           adc_data_in0;
    wire busy0, done0;

    optical_bridge_axi_lite #(
        .N(N0), .C(C0), .DAC_WIDTH(DACW0), .ADC_WIDTH(ADCW0),
        .AXI_ADDR_WIDTH(AXI_AW0),
        .SETTLE_CYCLES_PLACEHOLDER(4), .ADC_LATENCY_CYCLES_PLACEHOLDER(2)
    ) dut_small (
        .s_axi_aclk(clk), .s_axi_aresetn(aresetn),
        .s_axi_awaddr(awaddr0), .s_axi_awvalid(awvalid0), .s_axi_awready(awready0),
        .s_axi_wdata(wdata0),   .s_axi_wstrb(4'hF), .s_axi_wvalid(wvalid0), .s_axi_wready(wready0),
        .s_axi_bresp(bresp0),   .s_axi_bvalid(bvalid0), .s_axi_bready(bready0),
        .s_axi_araddr(araddr0), .s_axi_arvalid(arvalid0), .s_axi_arready(arready0),
        .s_axi_rdata(rdata0),   .s_axi_rresp(rresp0), .s_axi_rvalid(rvalid0), .s_axi_rready(rready0),
        .dac_bus_addr(dac_addr0), .dac_bus_data(dac_data0), .dac_bus_wr_en(dac_wr0),
        .adc_bus_addr(adc_addr0), .adc_bus_rd_en(adc_rd0), .adc_bus_data_in(adc_data_in0),
        .busy(busy0), .done(done0)
    );

    // Placeholder "optical hardware stand-in": present a known, address-derived
    // value on the ADC bus. Formula is arbitrary -- it only exists so the
    // testbench can check the FSM latched the right value at the right index.
    always @(*) adc_data_in0 = (adc_addr0 * 17 + 3) & ((1 << ADCW0) - 1);

    // =====================================================================
    // Instance 2: module's own PLACEHOLDER default sizes (N=10, C=10)
    // =====================================================================
    localparam AXI_AW1 = 16;
    wire [AXI_AW1-1:0] dummy_addr_width_check; // unused, silences lint about AXI_AW1

    reg  [15:0] awaddr1; reg awvalid1; wire awready1;
    reg  [31:0] wdata1;  reg wvalid1;  wire wready1;
    wire [1:0]   bresp1; wire bvalid1; reg bready1;
    reg  [15:0] araddr1; reg arvalid1; wire arready1;
    wire [31:0]  rdata1; wire [1:0] rresp1; wire rvalid1; reg rready1;

    // N=10 -> 2N-1=19 DAC entries -> idx width 5; C=10 -> idx width 4
    wire [4:0] dac_addr1;
    wire [11:0] dac_data1;
    wire dac_wr1;
    wire [3:0] adc_addr1;
    wire adc_rd1;
    reg [11:0] adc_data_in1;
    wire busy1, done1;

    optical_bridge_axi_lite dut_default ( // all defaults: N=10, C=10, DAC/ADC width=12
        .s_axi_aclk(clk), .s_axi_aresetn(aresetn),
        .s_axi_awaddr(awaddr1), .s_axi_awvalid(awvalid1), .s_axi_awready(awready1),
        .s_axi_wdata(wdata1),   .s_axi_wstrb(4'hF), .s_axi_wvalid(wvalid1), .s_axi_wready(wready1),
        .s_axi_bresp(bresp1),   .s_axi_bvalid(bvalid1), .s_axi_bready(bready1),
        .s_axi_araddr(araddr1), .s_axi_arvalid(arvalid1), .s_axi_arready(arready1),
        .s_axi_rdata(rdata1),   .s_axi_rresp(rresp1), .s_axi_rvalid(rvalid1), .s_axi_rready(rready1),
        .dac_bus_addr(dac_addr1), .dac_bus_data(dac_data1), .dac_bus_wr_en(dac_wr1),
        .adc_bus_addr(adc_addr1), .adc_bus_rd_en(adc_rd1), .adc_bus_data_in(adc_data_in1),
        .busy(busy1), .done(done1)
    );

    always @(*) adc_data_in1 = (adc_addr1 * 17 + 3) & 12'hFFF;

    // =====================================================================
    // Generic AXI4-Lite master BFM tasks, parameterized over which
    // instance's signals to drive via a `sel` bit (0 = small, 1 = default).
    // Written once, reused for both instances to avoid duplicating the
    // handshake logic.
    // =====================================================================
    reg sel; // 0 = dut_small signals, 1 = dut_default signals

    task axi_write(input [15:0] addr, input [31:0] data);
        integer timeout;
        begin
            if (sel == 1'b0) begin
                awaddr0 = addr; wdata0 = data;
                awvalid0 = 1'b1; wvalid0 = 1'b1;
            end else begin
                awaddr1 = addr; wdata1 = data;
                awvalid1 = 1'b1; wvalid1 = 1'b1;
            end

            // Wait for AW accept
            timeout = 0;
            if (sel == 1'b0) begin
                while (!awready0 && timeout < 100) begin @(posedge clk); timeout = timeout + 1; end
            end else begin
                while (!awready1 && timeout < 100) begin @(posedge clk); timeout = timeout + 1; end
            end
            @(posedge clk); #1;
            if (sel == 1'b0) awvalid0 = 1'b0; else awvalid1 = 1'b0;

            // Wait for W accept
            timeout = 0;
            if (sel == 1'b0) begin
                while (!wready0 && timeout < 100) begin @(posedge clk); timeout = timeout + 1; end
            end else begin
                while (!wready1 && timeout < 100) begin @(posedge clk); timeout = timeout + 1; end
            end
            @(posedge clk); #1;
            if (sel == 1'b0) wvalid0 = 1'b0; else wvalid1 = 1'b0;

            // Response phase
            if (sel == 1'b0) bready0 = 1'b1; else bready1 = 1'b1;
            timeout = 0;
            if (sel == 1'b0) begin
                while (!bvalid0 && timeout < 100) begin @(posedge clk); timeout = timeout + 1; end
            end else begin
                while (!bvalid1 && timeout < 100) begin @(posedge clk); timeout = timeout + 1; end
            end
            @(posedge clk); #1;
            if (sel == 1'b0) bready0 = 1'b0; else bready1 = 1'b0;

            if (timeout >= 100) begin
                $display("[TB] AXI WRITE TIMEOUT addr=0x%04h", addr);
                errors = errors + 1;
            end
        end
    endtask

    task axi_read(input [15:0] addr, output [31:0] data);
        integer timeout;
        begin
            if (sel == 1'b0) begin
                araddr0 = addr; arvalid0 = 1'b1;
            end else begin
                araddr1 = addr; arvalid1 = 1'b1;
            end

            timeout = 0;
            if (sel == 1'b0) begin
                while (!arready0 && timeout < 100) begin @(posedge clk); timeout = timeout + 1; end
            end else begin
                while (!arready1 && timeout < 100) begin @(posedge clk); timeout = timeout + 1; end
            end
            @(posedge clk); #1;
            if (sel == 1'b0) arvalid0 = 1'b0; else arvalid1 = 1'b0;

            if (sel == 1'b0) rready0 = 1'b1; else rready1 = 1'b1;
            timeout = 0;
            if (sel == 1'b0) begin
                while (!rvalid0 && timeout < 100) begin @(posedge clk); timeout = timeout + 1; end
            end else begin
                while (!rvalid1 && timeout < 100) begin @(posedge clk); timeout = timeout + 1; end
            end
            data = (sel == 1'b0) ? rdata0 : rdata1;
            @(posedge clk); #1;
            if (sel == 1'b0) rready0 = 1'b0; else rready1 = 1'b0;

            if (timeout >= 100) begin
                $display("[TB] AXI READ TIMEOUT addr=0x%04h", addr);
                errors = errors + 1;
            end
        end
    endtask

    task check_equal(input [63:0] got, input [63:0] expect_, input [255:0] label);
        begin
            if (got !== expect_) begin
                $display("[TB] FAIL %0s : got=0x%0h expect=0x%0h", label, got, expect_);
                errors = errors + 1;
            end else begin
                $display("[TB] pass %0s : 0x%0h", label, got);
            end
        end
    endtask

    // =====================================================================
    // DAC/ADC bus sequence monitors (independent of the AXI side -- these
    // watch the placeholder physical-side ports directly to confirm the
    // FSM visits every index exactly once, in order).
    // =====================================================================
    integer dac_seq_count0, adc_seq_count0, dac_seq_errors0, adc_seq_errors0;
    integer dac_seq_count1, adc_seq_count1, dac_seq_errors1, adc_seq_errors1;

    task monitor_reset_small; begin dac_seq_count0=0; adc_seq_count0=0; dac_seq_errors0=0; adc_seq_errors0=0; end endtask
    task monitor_reset_default; begin dac_seq_count1=0; adc_seq_count1=0; dac_seq_errors1=0; adc_seq_errors1=0; end endtask

    always @(posedge clk) begin
        if (dac_wr0) begin
            if (dac_addr0 !== dac_seq_count0[$bits(dac_addr0)-1:0])
                dac_seq_errors0 = dac_seq_errors0 + 1;
            dac_seq_count0 = dac_seq_count0 + 1;
        end
        if (adc_rd0) begin
            if (adc_addr0 !== adc_seq_count0[$bits(adc_addr0)-1:0])
                adc_seq_errors0 = adc_seq_errors0 + 1;
            adc_seq_count0 = adc_seq_count0 + 1;
        end
        if (dac_wr1) begin
            if (dac_addr1 !== dac_seq_count1[$bits(dac_addr1)-1:0])
                dac_seq_errors1 = dac_seq_errors1 + 1;
            dac_seq_count1 = dac_seq_count1 + 1;
        end
        if (adc_rd1) begin
            if (adc_addr1 !== adc_seq_count1[$bits(adc_addr1)-1:0])
                adc_seq_errors1 = adc_seq_errors1 + 1;
            adc_seq_count1 = adc_seq_count1 + 1;
        end
    end

    // =====================================================================
    // Register map constants mirrored here for the testbench's own use
    // =====================================================================
    localparam [15:0] A_CTRL    = 16'h0000;
    localparam [15:0] A_STATUS  = 16'h0004;
    localparam [15:0] A_VERSION = 16'h0008;
    localparam [15:0] A_DAC0    = 16'h0100;
    localparam [15:0] A_ADC0    = 16'h0400;

    reg [31:0] rd;
    integer i;
    integer poll_timeout;

    task run_full_sequence_check(
        input integer n_dac, input integer n_adc, input [255:0] tag
    );
        begin
            // 1. VERSION sanity
            axi_read(A_VERSION, rd);
            check_equal(rd, 32'hCAFEBEEF, {tag, " VERSION"});

            // 2. Write known pattern into every DAC entry, read back
            for (i = 0; i < n_dac; i = i + 1)
                axi_write(A_DAC0 + 4*i, 32'h100 + i);
            for (i = 0; i < n_dac; i = i + 1) begin
                axi_read(A_DAC0 + 4*i, rd);
                check_equal(rd, 32'h100 + i, {tag, " DAC readback"});
            end

            // 3. Confirm BUSY=0, DONE=0 before start
            axi_read(A_STATUS, rd);
            check_equal(rd[0], 1'b0, {tag, " BUSY before start"});

            // 4. Trigger sequence
            if (sel == 1'b0) begin monitor_reset_small; end else begin monitor_reset_default; end
            axi_write(A_CTRL, 32'h1);

            // 5. Poll for DONE (bounded wait)
            poll_timeout = 0;
            rd = 32'h0;
            while (rd[1] !== 1'b1 && poll_timeout < 500) begin
                axi_read(A_STATUS, rd);
                poll_timeout = poll_timeout + 1;
            end
            if (poll_timeout >= 500) begin
                $display("[TB] FAIL %0s : sequence never completed (DONE never set)", tag);
                errors = errors + 1;
            end else begin
                $display("[TB] pass %0s : sequence completed in <= %0d AXI poll reads", tag, poll_timeout);
            end
            check_equal(rd[0], 1'b0, {tag, " BUSY after done"});

            // 6. Confirm the dac_bus/adc_bus monitors saw exactly the right
            //    number of strobes, in order, with no duplicates/skips.
            if (sel == 1'b0) begin
                check_equal(dac_seq_count0, n_dac, {tag, " dac_bus strobe count"});
                check_equal(adc_seq_count0, n_adc, {tag, " adc_bus strobe count"});
                check_equal(dac_seq_errors0, 0, {tag, " dac_bus sequencing errors"});
                check_equal(adc_seq_errors0, 0, {tag, " adc_bus sequencing errors"});
            end else begin
                check_equal(dac_seq_count1, n_dac, {tag, " dac_bus strobe count"});
                check_equal(adc_seq_count1, n_adc, {tag, " adc_bus strobe count"});
                check_equal(dac_seq_errors1, 0, {tag, " dac_bus sequencing errors"});
                check_equal(adc_seq_errors1, 0, {tag, " adc_bus sequencing errors"});
            end

            // 7. Confirm ADC_TABLE was latched with the expected
            //    address-derived placeholder-hardware values.
            for (i = 0; i < n_adc; i = i + 1) begin
                axi_read(A_ADC0 + 4*i, rd);
                check_equal(rd, (i*17+3) & 32'hFFF, {tag, " ADC readback"});
            end
        end
    endtask

    // =====================================================================
    // Main test sequence
    // =====================================================================
    initial begin
        awvalid0=0; wvalid0=0; bready0=0; arvalid0=0; rready0=0;
        awvalid1=0; wvalid1=0; bready1=0; arvalid1=0; rready1=0;
        monitor_reset_small; monitor_reset_default;

        aresetn = 1'b0;
        repeat (5) @(posedge clk);
        aresetn = 1'b1;
        repeat (3) @(posedge clk);

        $display("================================================================");
        $display("[TB] optical_bridge_axi_lite -- PRELIMINARY/SPECULATIVE sketch");
        $display("[TB] simulation-only, no real DAC/ADC/photonic hardware involved");
        $display("================================================================");

        // ---- Instance 1: small sizes (N=3 -> 5 DAC entries, C=2) ----
        sel = 1'b0;
        $display("[TB] --- dut_small (N=%0d, C=%0d) : run 1 ---", N0, C0);
        run_full_sequence_check(2*N0-1, C0, "small-run1");

        $display("[TB] --- dut_small (N=%0d, C=%0d) : run 2 (re-trigger, no reset) ---", N0, C0);
        run_full_sequence_check(2*N0-1, C0, "small-run2");

        // ---- Instance 2: module's own PLACEHOLDER default sizes (N=10, C=10) ----
        sel = 1'b1;
        $display("[TB] --- dut_default (N=10, C=10, module defaults) : run 1 ---");
        run_full_sequence_check(19, 10, "default-run1");

        $display("[TB] --- dut_default (N=10, C=10, module defaults) : run 2 (re-trigger) ---");
        run_full_sequence_check(19, 10, "default-run2");

        $display("================================================================");
        if (errors == 0) begin
            $display("[TB] RESULT: ALL CHECKS PASSED (%0d errors)", errors);
        end else begin
            $display("[TB] RESULT: %0d CHECK(S) FAILED", errors);
        end
        $display("================================================================");
        $finish;
    end

    // Safety watchdog in case something hangs completely
    initial begin
        #200000;
        $display("[TB] WATCHDOG TIMEOUT -- simulation did not finish in time");
        $finish;
    end

endmodule
