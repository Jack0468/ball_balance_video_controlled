#!/bin/bash
#
# install_l4t_prereqs.sh
#
# Installs the .deb packages required by l4t_flash_prerequisites.sh
# from a local folder, with NO internet connection required.
#
# Usage:
#   1. Copy all downloaded .deb files into a single folder
#      (e.g. the folder you copied off your USB drive).
#   2. Run this script, pointing it at that folder:
#        sudo ./install_l4t_prereqs.sh /path/to/deb-folder
#      or just run it from inside the folder with no argument:
#        cd /path/to/deb-folder && sudo /path/to/install_l4t_prereqs.sh
#
# What it does:
#   - Checks which of the required packages are already installed
#     (skips those, no need to reinstall)
#   - Installs the .deb files found in the target folder for anything missing
#   - Tries to fix any remaining unmet dependencies using ONLY local
#     packages already present in the folder (--no-download), so it
#     never tries to reach the internet
#   - Reports at the end exactly what's still missing, if anything
#
set -u

# ---- Config: the full list of required packages ----
REQUIRED_PACKAGES=(
  lbzip2
  qemu-user-static
  python3-minimal
  python3-yaml
  cpio
  openssh-client
  binutils
  xxd
  dosfstools
  perl
  lz4
  libxml2-utils
  sudo
  xz-utils
  cpp
  whiptail
  openssl
  dpkg-dev
)

# ---- Determine the folder containing the .deb files ----
DEB_DIR="${1:-$(pwd)}"

if [[ ! -d "$DEB_DIR" ]]; then
  echo "ERROR: '$DEB_DIR' is not a valid directory." >&2
  exit 1
fi

# ---- Must be run as root (installing packages) ----
if [[ "$EUID" -ne 0 ]]; then
  echo "This script needs to install packages system-wide, so please re-run it with sudo:"
  echo "  sudo $0 $DEB_DIR"
  exit 1
fi

echo "==> Using .deb folder: $DEB_DIR"
echo

# ---- Step 1: Check what's already installed ----
echo "==> Checking which required packages are already installed..."
MISSING_PACKAGES=()
for pkg in "${REQUIRED_PACKAGES[@]}"; do
  if dpkg -s "$pkg" >/dev/null 2>&1; then
    echo "    [already installed] $pkg"
  else
    echo "    [missing]           $pkg"
    MISSING_PACKAGES+=("$pkg")
  fi
done
echo

if [[ ${#MISSING_PACKAGES[@]} -eq 0 ]]; then
  echo "All required packages are already installed. Nothing to do."
  exit 0
fi

# ---- Step 2: Find matching .deb files in the folder for missing packages ----
echo "==> Looking for matching .deb files in $DEB_DIR ..."
DEBS_TO_INSTALL=()
STILL_MISSING=()

for pkg in "${MISSING_PACKAGES[@]}"; do
  # Match files like: pkgname_1.2.3-1_amd64.deb  (allowing for version/arch variations)
  match=$(find "$DEB_DIR" -maxdepth 1 -iname "${pkg}_*.deb" | head -n 1)
  if [[ -n "$match" ]]; then
    echo "    found: $(basename "$match")"
    DEBS_TO_INSTALL+=("$match")
  else
    echo "    NOT FOUND: no .deb file matching '${pkg}_*.deb' in $DEB_DIR"
    STILL_MISSING+=("$pkg")
  fi
done
echo

if [[ ${#DEBS_TO_INSTALL[@]} -eq 0 ]]; then
  echo "No matching .deb files found for any missing package. Nothing to install."
  echo "Missing packages: ${STILL_MISSING[*]}"
  exit 1
fi

# ---- Step 3: Install the .deb files ----
echo "==> Installing ${#DEBS_TO_INSTALL[@]} package(s) with dpkg..."
dpkg -i "${DEBS_TO_INSTALL[@]}"
DPKG_STATUS=$?
echo

# ---- Step 4: Fix any broken/unmet dependencies, using ONLY local files ----
# --no-download ensures apt-get never tries to reach the internet.
# It will only succeed if the needed dependency .debs are also in $DEB_DIR
# and have already been placed in apt's local cache via dpkg -i above,
# or are available via `apt-get install -f` using packages already unpacked.
if [[ $DPKG_STATUS -ne 0 ]]; then
  echo "==> dpkg reported issues (likely unmet dependencies). Attempting local-only fix..."
  apt-get install -f -y --no-download
  echo
fi

# ---- Step 5: Final verification ----
echo "==> Final check..."
FINAL_MISSING=()
for pkg in "${REQUIRED_PACKAGES[@]}"; do
  if ! dpkg -s "$pkg" >/dev/null 2>&1; then
    FINAL_MISSING+=("$pkg")
  fi
done

echo
if [[ ${#FINAL_MISSING[@]} -eq 0 ]]; then
  echo "SUCCESS: all required packages are now installed."
else
  echo "WARNING: the following packages are still missing or failed to install:"
  for pkg in "${FINAL_MISSING[@]}"; do
    echo "    - $pkg"
  done
  echo
  echo "This usually means a dependency .deb is missing from $DEB_DIR."
  echo "Run the following to see exactly what's unmet (no internet needed, just diagnostic):"
  echo "    apt-get install -f --no-download --dry-run"
  echo
  echo "Download the missing dependency .deb file(s) named above, place them in"
  echo "$DEB_DIR, and re-run this script."
fi
