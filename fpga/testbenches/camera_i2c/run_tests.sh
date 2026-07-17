#!/bin/bash

# Ensure we're in the right directory
cd "$(dirname "$0")"

echo "Compiling SDRAM Arbiter Testbench..."
iverilog -o tb_sdram_arbiter.vvp \
    tb_sdram_arbiter.v \
    ../../camera_i2c/sdram_arbiter.v \
    ../../camera_i2c/fifo.v \
    ../../camera_i2c/sdram_stffrdhrn/rtl/sdram_controller.v

if [ $? -eq 0 ]; then
    echo "Compilation successful. Running simulation..."
    vvp tb_sdram_arbiter.vvp
else
    echo "Compilation failed."
fi
