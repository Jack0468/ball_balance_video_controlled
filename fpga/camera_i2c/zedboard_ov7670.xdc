# ==============================================================================
# OV7670 Camera Constraints for ZedBoard (Zynq-7000 XC7Z020)
# ==============================================================================

# 1. Clock Definition
# The OV7670 outputs PCLK at 24MHz (41.667 ns period). 
# We explicitly define this so Vivado's timing engine can analyze setup/hold times.
create_clock -period 41.667 -name pclk [get_ports pclk]

# Bypass Vivado's strict IO-to-BUFG routing rules since PMOD pins aren't 
# always in the ideal clock region, and our 24MHz clock is slow enough that 
# the routing skew is negligible.
set_property CLOCK_DEDICATED_ROUTE FALSE [get_nets pclk_IBUF]

# ==============================================================================
# PMOD JC (Bank 13 - 3.3V)
# ==============================================================================

# PCLK MUST be on a Clock-Capable pin. 
# JC1_P (AB7) is a Multi-Region Clock Capable (MRCC) pin on the Zynq 7020!
set_property -dict { PACKAGE_PIN AB7  IOSTANDARD LVCMOS33 } [get_ports pclk];  # JC Pin 1
set_property -dict { PACKAGE_PIN AB6  IOSTANDARD LVCMOS33 } [get_ports xclk];  # JC Pin 2
set_property -dict { PACKAGE_PIN Y4   IOSTANDARD LVCMOS33 } [get_ports vsync]; # JC Pin 3
set_property -dict { PACKAGE_PIN AA4  IOSTANDARD LVCMOS33 } [get_ports href];  # JC Pin 4

set_property -dict { PACKAGE_PIN R6   IOSTANDARD LVCMOS33 } [get_ports sioc];  # JC Pin 7
set_property -dict { PACKAGE_PIN T6   IOSTANDARD LVCMOS33 } [get_ports siod];  # JC Pin 8
set_property -dict { PACKAGE_PIN T4   IOSTANDARD LVCMOS33 } [get_ports {p_data[7]}]; # JC Pin 9
set_property -dict { PACKAGE_PIN U4   IOSTANDARD LVCMOS33 } [get_ports {p_data[6]}]; # JC Pin 10

# ==============================================================================
# PMOD JD (Bank 13 - 3.3V)
# ==============================================================================

set_property -dict { PACKAGE_PIN V7   IOSTANDARD LVCMOS33 } [get_ports {p_data[5]}]; # JD Pin 1
set_property -dict { PACKAGE_PIN W7   IOSTANDARD LVCMOS33 } [get_ports {p_data[4]}]; # JD Pin 2
set_property -dict { PACKAGE_PIN V5   IOSTANDARD LVCMOS33 } [get_ports {p_data[3]}]; # JD Pin 3
set_property -dict { PACKAGE_PIN V4   IOSTANDARD LVCMOS33 } [get_ports {p_data[2]}]; # JD Pin 4

set_property -dict { PACKAGE_PIN W6   IOSTANDARD LVCMOS33 } [get_ports {p_data[1]}]; # JD Pin 7
set_property -dict { PACKAGE_PIN W5   IOSTANDARD LVCMOS33 } [get_ports {p_data[0]}]; # JD Pin 8
# JD Pin 9 (U6) and Pin 10 (U5) are left completely unused, or you can use them for RESET/PWDN if you prefer not to hardwire them.

# Pullups for I2C (SCCB) lines
set_property PULLUP true [get_ports sioc]
set_property PULLUP true [get_ports siod]
