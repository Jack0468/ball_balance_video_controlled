# Jetson AGX Orin 64GB — Offline Flash Project: Handoff Summary

## Objective
Flash a **Jetson AGX Orin Developer Kit (64GB)** with a basic Ubuntu OS using an
**offline Ubuntu host machine** (no internet access, no built-in WiFi hardware),
via manual L4T tools (not SDK Manager, which doesn't support this use case
reliably). After the base flash, a **Windows machine** will be used separately
to install JetPack SDK components (CUDA, cuDNN, TensorRT, etc.) over the
network via SDK Manager, without re-flashing the OS.

## Hardware / Software Target
- **Board:** Jetson AGX Orin Developer Kit, 64GB (module P3701-0000/P3701-0005 family)
- **Flash config name:** `jetson-agx-orin-devkit`
- **Flash target argument:** `internal` (current syntax for r36.x; NOT the legacy `mmcblk0p1`)
- **L4T release:** r36.5.2 (part of JetPack 6.2.3), Ubuntu 22.04 (jammy) base, Linux Kernel 5.15
- **Host machine:** Ubuntu (amd64/x86_64), **no internet access**, no WiFi hardware
- **Recovery mode USB ID:** `0955:7023` (confirmed via NVIDIA's official Quick Start doc)
- **Correct flashing USB-C port:** the one **next to the 40-pin header** (not the other USB-C port)

## Files Downloaded (from developer.nvidia.com/embedded/jetson-linux-r3652)
1. `Jetson_Linux_r36.5.2_aarch64.tbz2` (Driver Package / BSP)
2. `Tegra_Linux_Sample-Root-Filesystem_r36.5.2_aarch64.tbz2` (Sample Root Filesystem)

Both downloaded from an internet-connected device (not necessarily Ubuntu),
transferred to the offline Ubuntu host via a USB drive formatted exFAT.

## Method: Offline "Sneakernet" Workflow
Since the flashing host has no internet:
1. Download files/packages on any internet-connected device (Windows used in practice).
2. Copy to a USB drive (exFAT format, for cross-OS compatibility).
3. Transfer to the offline Ubuntu host.
4. Install `.deb` packages locally via `dpkg -i`, never touching the network.

## Two Custom Scripts Built

### 1. `install_l4t_prereqs.sh`
Installs required `.deb` packages from a local folder, entirely offline.
- Checks which required packages are already installed (skips those)
- Matches missing ones to `.deb` files in a given folder by filename pattern
- Installs via `dpkg -i`, then attempts `apt-get install -f --no-download` to
  fix any unmet dependencies **using only local files** (never touches network)
- Reports final pass/fail status per package

Usage: `sudo ./install_l4t_prereqs.sh /path/to/deb-folder`

### 2. `flash_agx_orin.sh`
Wrapper script that automates the extraction + flash process, calling
NVIDIA's own official tools (does not reimplement flashing logic):
1. Verifies amd64 host + both tarballs present
2. Extracts BSP (`tar xf`) → creates `Linux_for_Tegra/`
3. Extracts Sample Root Filesystem into `rootfs/` (`sudo tar xpf`)
4. Runs `sudo ./apply_binaries.sh` (order matters — confirmed against NVIDIA's
   official Quick Start guide: apply_binaries.sh runs BEFORE the prerequisites check)
5. Runs `sudo ./tools/l4t_flash_prerequisites.sh`
6. Pauses with on-screen Force Recovery Mode instructions, waits for Enter
7. Checks `lsusb` for ID `0955:7023` to confirm recovery mode before proceeding
8. Runs `sudo ./flash.sh jetson-agx-orin-devkit internal`

Script is idempotent for steps 2–4 (uses directory existence / a
`.apply_binaries_done` marker file to skip already-completed steps on re-run).

Usage: `sudo ./flash_agx_orin.sh` (run from the folder containing both tarballs)

## Required Offline `.deb` Packages — Full List (as discovered so far)

**Original baseline list** (sourced from NVIDIA's official Jetson Linux Flash
container Docker build layers on catalog.ngc.nvidia.com):
```
lbzip2, qemu-user-static, python3-minimal, python3-yaml, cpio,
openssh-client, binutils, xxd, dosfstools, perl, lz4, libxml2-utils,
sudo, xz-utils, cpp, whiptail, openssl, dpkg-dev
```

**Additional packages required by `l4t_flash_prerequisites.sh` for r36.5.2**
(discovered via actual script execution — NOT in the container build list):
```
abootimg, binfmt-support, device-tree-compiler, nfs-kernel-server,
sshpass, whois, xmlstarlet, python-is-python3
```

**Sub-dependencies discovered during install (transitive):**
- `binutils` → `binutils-common`, `libbinutils`, `binutils-x86-64-linux-gnu`
- `lz4` → (resolved cleanly, no extra deps needed)
- `device-tree-compiler` → `libfdt1` (>= 1.6.1)
- `nfs-kernel-server` → `libevent-core-2.1-7`, `nfs-common`, `keyutils`
- `nfs-common` → `libnfsidmap1`, `rpcbind`
- `xmlstarlet`, `python-is-python3` → requested again after being missed in an
  earlier download pass (may simply have been skipped by mistake, not a new
  dependency discovery)

**⚠️ Status as of last message:** dependency discovery for `nfs-kernel-server`
/ `nfs-common` / `rpcbind` chain may not be fully exhausted yet. `rpcbind`
itself may pull in further packages (e.g. `libtirpc3`, `libwrap0`) not yet
confirmed. **Recommended next step for whoever picks this up:** run the
following on the offline host to get a complete, authoritative list of
everything still outstanding in one pass, rather than continuing to discover
dependencies one at a time:
```bash
sudo apt-get install -f --no-download --dry-run
```

## Key Corrections Made Along the Way (verified against NVIDIA's official docs)
1. Flash order: `apply_binaries.sh` **before** `l4t_flash_prerequisites.sh` (not the reverse)
2. Flash target argument: `internal` (not legacy `mmcblk0p1`)
3. Recovery mode button sequence: power off → hold Force Recovery → press &
   release Power → release Force Recovery (no extra "wait" steps)
4. Filename case sensitivity: NVIDIA's actual download filenames may differ in
   capitalization from what's assumed — always verify with `ls -la *.tbz2`
   before running the flash script, and update the script's hardcoded
   filenames to match exactly if needed

## Issue Resolved: QEMU/binfmt registration failure
**Symptom:** `apply_binaries.sh` failed with `chroot: failed to run command
'dpkg': Exec format error` during the "Installing QEMU binary in rootfs" step.

**Cause:** `qemu-user-static` was installed, but the kernel's `binfmt_misc`
system was never registered to route ARM64 binaries through QEMU emulation
during chroot operations.

**Fix applied:**
```bash
sudo su
echo ':qemu-aarch64:M::\x7fELF\x02\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00\xb7\x00:\xff\xff\xff\xff\xff\xff\xff\x00\xff\xff\xff\xff\xff\xff\xff\xff\xfe\xff\xff\xff:/usr/bin/qemu-aarch64-static:CF' > /proc/sys/fs/binfmt_misc/register
exit
```
Verified with: `cat /proc/sys/fs/binfmt_misc/qemu-aarch64` (should show `enabled`)

**Result:** `apply_binaries.sh` subsequently completed successfully in full —
confirmed by `"L4T BSP package installation completed!"` / `"Success!"` in the log.

Note: installing `binfmt-support` (now in the required package list) should
make this registration persistent/automatic on future runs, reducing reliance
on this manual fix.

## Current Status (as of handoff)
- ✅ BSP and rootfs downloaded and extracted successfully
- ✅ `apply_binaries.sh` completed successfully
- 🔄 **IN PROGRESS:** `l4t_flash_prerequisites.sh` still failing due to an
  evolving list of missing offline `.deb` dependencies (see list above) —
  most recently `xmlstarlet` and `python-is-python3` were reported missing
  again (possibly a download omission rather than a new dependency)
- ⬜ Not yet reached: Force Recovery Mode step
- ⬜ Not yet reached: actual `flash.sh` execution
- ⬜ Not yet reached: first-boot OEM setup (requires physical monitor/keyboard
  on the Jetson — no headless/pre-seeded account has been configured; this
  was discussed but not implemented — see "Deferred Items" below)
- ⬜ Not yet reached: Windows machine / SDK Manager follow-up phase

## Deferred / Discussed but Not Implemented
- **Headless first-boot setup:** NVIDIA's `l4t_create_default_user.sh`
  (located at `Linux_for_Tegra/tools/l4t_create_default_user.sh`) can
  pre-create a user account in the rootfs before flashing, entirely
  skipping the OEM wizard and enabling immediate SSH access post-flash with
  no monitor needed. This was discussed as an option but the user chose to
  continue with the in-progress flash rather than restart with this change.
  **Worth considering for whoever continues this work**, especially if a
  monitor/keyboard on the physical Jetson is inconvenient.
- **Windows/SDK Manager follow-up phase:** confirmed feasible via NVIDIA's
  official docs — SDK Manager on Windows auto-provisions a WSL2 Ubuntu
  environment. Requires JetPack 6.2.1+ (r36.5.2 = JetPack 6.2.3, so
  compatible). Process: connect Jetson to network (Ethernet or USB, appears
  as ~192.168.55.1 for USB), run SDK Manager on Windows, deselect "Jetson
  OS" step, select "Jetson SDK Components" only, provide the Jetson's IP/
  username/password over SSH. Not yet started — the base OS flash must
  complete first.

## Immediate Next Steps for Continuation
1. Run `sudo apt-get install -f --no-download --dry-run` on the offline host
   to get the complete outstanding dependency list in one pass.
2. Download any remaining `.deb`s (including sub-dependencies — check each
   package's "Depends" section on packages.ubuntu.com before leaving the
   internet-connected machine, to avoid further round trips).
3. Install via `sudo dpkg -i prereq/*.deb` then `sudo apt-get install -f --no-download`.
4. Re-run `sudo ./flash_agx_orin.sh` — it will skip already-completed
   extraction/apply_binaries steps and re-check prerequisites.
5. Once prerequisites pass, follow the Force Recovery Mode prompt exactly as
   printed by the script, connect via the correct USB-C port, and let the
   flash complete (10–20 minutes, do not disconnect).
6. Complete first-boot OEM setup locally (monitor + keyboard required, unless
   `l4t_create_default_user.sh` is used beforehand — see Deferred Items).
7. Proceed to Windows/SDK Manager phase for SDK component installation.
