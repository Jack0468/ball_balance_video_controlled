import re

with open('user_orig_rom.txt', 'r') as f:
    lines = f.readlines()

start = False
rom_lines = []
for line in lines:
    line = line.rstrip()
    if line.startswith('+'):
        line = line[1:]
    
    if "0:  dout <= 16'h12_80; //reset" in line:
        start = True
    if start:
        rom_lines.append(line)
        if "default: dout <= 16'hFF_FF;" in line:
            break

module_code = '''`timescale 1ns / 1ps

module OV7670_config_rom(
    input wire clk,
    input wire [7:0] addr,
    output reg [15:0] dout
    );
    //FFFF is end of rom, FFF0 is delay
    always @(posedge clk) begin
    case(addr) 
'''

module_code += '\n'.join(rom_lines)

module_code += '''
    endcase
    
    end
endmodule
'''

with open('OV7670_config_rom.v', 'w') as f:
    f.write(module_code)
