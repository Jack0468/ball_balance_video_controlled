`timescale 1ns / 1ps

module fifo #(
    parameter DATA_WIDTH = 16,
    parameter ADDR_WIDTH = 11 // 2048 words
) (
    input  wire                  rst,
    
    input  wire                  wr_clk,
    input  wire                  wr_en,
    input  wire [DATA_WIDTH-1:0] din,
    output wire                  full,

    input  wire                  rd_clk,
    input  wire                  rd_en,
    output wire [DATA_WIDTH-1:0] dout,
    output wire                  empty,
    
    output wire                  overflow,
    output wire [ADDR_WIDTH-1:0] rd_data_count
);

    // Memory array
    reg [DATA_WIDTH-1:0] mem [0:(1<<ADDR_WIDTH)-1];

    // Pointers
    reg [ADDR_WIDTH:0] wr_ptr = 0;
    reg [ADDR_WIDTH:0] rd_ptr = 0;

    // Gray code pointers
    wire [ADDR_WIDTH:0] wr_ptr_gray = wr_ptr ^ (wr_ptr >> 1);
    wire [ADDR_WIDTH:0] rd_ptr_gray = rd_ptr ^ (rd_ptr >> 1);

    // Synchronized pointers
    reg [ADDR_WIDTH:0] wr_ptr_gray_sync1 = 0, wr_ptr_gray_sync2 = 0;
    reg [ADDR_WIDTH:0] rd_ptr_gray_sync1 = 0, rd_ptr_gray_sync2 = 0;

    // Write domain logic
    always @(posedge wr_clk or posedge rst) begin
        if (rst) begin
            wr_ptr <= 0;
            rd_ptr_gray_sync1 <= 0;
            rd_ptr_gray_sync2 <= 0;
        end else begin
            rd_ptr_gray_sync1 <= rd_ptr_gray;
            rd_ptr_gray_sync2 <= rd_ptr_gray_sync1;
            
            if (wr_en && !full) begin
                mem[wr_ptr[ADDR_WIDTH-1:0]] <= din;
                wr_ptr <= wr_ptr + 1;
            end
        end
    end

    // Read domain logic
    reg [DATA_WIDTH-1:0] rd_data_reg = 0;
    assign dout = rd_data_reg;

    always @(posedge rd_clk or posedge rst) begin
        if (rst) begin
            rd_ptr <= 0;
            wr_ptr_gray_sync1 <= 0;
            wr_ptr_gray_sync2 <= 0;
            rd_data_reg <= 0;
        end else begin
            wr_ptr_gray_sync1 <= wr_ptr_gray;
            wr_ptr_gray_sync2 <= wr_ptr_gray_sync1;
            
            if (rd_en && !empty) begin
                rd_data_reg <= mem[rd_ptr[ADDR_WIDTH-1:0]];
                rd_ptr <= rd_ptr + 1;
            end
        end
    end

    // Empty and Full flags (using gray code comparisons)
    // For full, the MSB and MSB-1 are inverted, the rest are the same
    wire [ADDR_WIDTH:0] wr_ptr_gray_next = (wr_ptr + 1) ^ ((wr_ptr + 1) >> 1);
    
    assign full = (wr_ptr_gray_next == {~rd_ptr_gray_sync2[ADDR_WIDTH:ADDR_WIDTH-1], rd_ptr_gray_sync2[ADDR_WIDTH-2:0]});
    assign empty = (rd_ptr_gray == wr_ptr_gray_sync2);

    // Overflow logic
    reg overflow_reg = 0;
    always @(posedge wr_clk or posedge rst) begin
        if (rst) overflow_reg <= 0;
        else if (wr_en && full) overflow_reg <= 1;
    end
    assign overflow = overflow_reg;

    // Data count logic (Read domain)
    function [ADDR_WIDTH:0] gray_to_bin;
        input [ADDR_WIDTH:0] gray;
        integer i;
        begin
            gray_to_bin[ADDR_WIDTH] = gray[ADDR_WIDTH];
            for (i = ADDR_WIDTH-1; i >= 0; i = i - 1) begin
                gray_to_bin[i] = gray_to_bin[i+1] ^ gray[i];
            end
        end
    endfunction
    
    wire [ADDR_WIDTH:0] wr_ptr_bin_sync = gray_to_bin(wr_ptr_gray_sync2);
    wire [ADDR_WIDTH:0] diff = wr_ptr_bin_sync - rd_ptr;
    assign rd_data_count = diff[ADDR_WIDTH-1:0];

endmodule
