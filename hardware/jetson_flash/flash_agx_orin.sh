#!/bin/bash
#
# flash_agx_orin.sh
#
# Wrapper script to prepare and flash a Jetson AGX Orin 64GB Developer Kit
# using NVIDIA's official Linux_for_Tegra tools (BSP + rootfs already
# downloaded and extracted on this host).
#
# This does NOT replace NVIDIA's flash.sh — it automates the setup steps
# around it (extraction, prerequisites, apply_binaries, recovery-mode
# checks) and then calls the official flash.sh with the correct config
# for the 64GB AGX Orin Developer Kit.
#
# Usage:
#   1. Put these two files in the same directory as this script:
#        Jetson_Linux_r36.5.2_aarch64.tbz2
#        Tegra_Linux_Sample-Root-Filesystem_r36.5.2_aarch64.tbz2
#   2. Run:
#        sudo ./flash_agx_orin.sh
#   3. When prompted, put the AGX Orin into Force Recovery Mode and
#      connect it via USB-C to this host, then press Enter.
#
set -euo pipefail

# ---- Config ----
WORKDIR="$(pwd)/jetson-flash"
L4T_RELEASE_PACKAGE="Jetson_Linux_r36.5.2_aarch64.tbz2"
SAMPLE_FS_PACKAGE="Tegra_Linux_Sample-Root-Filesystem_r36.5.2_aarch64.tbz2"
BOARD_CONFIG="jetson-agx-orin-devkit"   # 64GB AGX Orin Developer Kit
STORAGE_DEVICE="internal"               # per NVIDIA's official r36.x Quick Start guide

# ---- Must be run as root ----
if [[ "$EUID" -ne 0 ]]; then
  echo "This script needs root privileges for flashing. Please re-run with sudo:"
  echo "  sudo $0"
  exit 1
fi

# ---- Sanity check: architecture ----
HOST_ARCH="$(dpkg --print-architecture)"
if [[ "$HOST_ARCH" != "amd64" ]]; then
  echo "ERROR: this must be run on an amd64 host (found: $HOST_ARCH)."
  exit 1
fi

# ---- Sanity check: required files present ----
for f in "$L4T_RELEASE_PACKAGE" "$SAMPLE_FS_PACKAGE"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: required file not found in current directory: $f"
    echo "Make sure both downloaded tarballs are in: $(pwd)"
    exit 1
  fi
done

echo "==> Files found. Preparing workspace at: $WORKDIR"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

# ---- Step 1: Extract the BSP (Driver Package) if not already done ----
if [[ ! -d "Linux_for_Tegra" ]]; then
  echo "==> Extracting Driver Package (BSP)..."
  tar xf "../$L4T_RELEASE_PACKAGE"
else
  echo "==> Linux_for_Tegra already extracted, skipping BSP extraction."
fi

cd Linux_for_Tegra

# ---- Step 2: Extract the sample rootfs into it, if not already done ----
if [[ ! -f "rootfs/etc/os-release" ]]; then
  echo "==> Extracting Sample Root Filesystem into rootfs/..."
  sudo tar xpf "../../$SAMPLE_FS_PACKAGE" -C rootfs/
else
  echo "==> Rootfs already populated, skipping rootfs extraction."
fi

# ---- Step 3: Apply binaries (NVIDIA firmware/drivers into rootfs) ----
# NVIDIA's official Quick Start guide runs apply_binaries.sh BEFORE
# l4t_flash_prerequisites.sh -- order matters, so this matches that.
if [[ ! -f ".apply_binaries_done" ]]; then
  echo "==> Applying NVIDIA binaries to rootfs (apply_binaries.sh)..."
  sudo ./apply_binaries.sh
  touch ".apply_binaries_done"
else
  echo "==> apply_binaries.sh already run, skipping."
fi

# ---- Step 4: Check flash prerequisites (offline-safe) ----
echo "==> Checking flash prerequisites (l4t_flash_prerequisites.sh)..."
if ! sudo ./tools/l4t_flash_prerequisites.sh; then
  echo
  echo "ERROR: prerequisite check failed."
  echo "If this is due to a missing package, install it offline first"
  echo "(see install_l4t_prereqs.sh from earlier), then re-run this script."
  exit 1
fi

# ---- Step 5: Prompt user to put the board into Force Recovery Mode ----
# Steps below are exactly as specified in NVIDIA's official r36.x Quick Start
# guide ("Put your Jetson developer kit into Force Recovery Mode").
echo
echo "=================================================================="
echo " Put the Jetson AGX Orin into FORCE RECOVERY MODE now:"
echo "   1. Ensure the developer kit is powered off."
echo "   2. Press and hold down the Force Recovery button."
echo "   3. Press, then release, the Power button."
echo "   4. Release the Force Recovery button."
echo
echo " Make sure the USB-C cable from the host is connected to the port"
echo " NEXT TO THE 40-PIN HEADER (this is the flashing port on the AGX"
echo " Orin Developer Kit -- not the port used for normal operation)."
echo "=================================================================="
read -rp "Press Enter once the board is in recovery mode and connected... "

# ---- Step 6: Verify the board is detected in recovery mode ----
# Per NVIDIA's official docs, the AGX Orin Developer Kit module (both the
# 32GB and 64GB variants) reports USB ID 0955:7023 when in Force Recovery
# Mode. This checks for that exact ID rather than any NVIDIA device.
echo "==> Checking for the board via lsusb..."
if lsusb | grep -qi "0955:7023"; then
  echo "    Board detected in Force Recovery Mode (ID 0955:7023)."
else
  echo "ERROR: expected device ID 0955:7023 not found on USB."
  echo "Current lsusb output:"
  lsusb
  echo
  echo "Check that: the cable is in the correct USB-C port (next to the"
  echo "40-pin header), the board is truly in recovery mode, and the cable"
  echo "itself supports data (not power-only). Then re-run this script."
  exit 1
fi

# ---- Step 7: Flash ----
echo
echo "==> Flashing Jetson AGX Orin 64GB Developer Kit..."
echo "    Config: $BOARD_CONFIG"
echo "    Target: $STORAGE_DEVICE"
echo
sudo ./flash.sh "$BOARD_CONFIG" "$STORAGE_DEVICE"

echo
echo "=================================================================="
echo " Flash complete. The board should reboot automatically."
echo " If it doesn't boot to the Ubuntu OEM setup screen, disconnect"
echo " power, reconnect, and power it on manually."
echo "=================================================================="
