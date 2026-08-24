# Jetson AGX Orin — Flash Procedure

Covers getting JetPack onto the Jetson AGX Orin 64GB Developer Kit (p3730). **Native
Ubuntu (dual-boot or Live USB) is the recommended path** — two earlier approaches
(VirtualBox VM, then WSL2 + `usbipd-win`) were both tried and both hit real,
board-identification-level failures; see "Why not VirtualBox" / "Why not WSL2" below for
what happened and why they're not recommended, kept for reference in case either is
revisited.

**Scope note**: this doc is the pre-boot flashing procedure only (getting the OS onto the
device). Post-boot dependency setup (Python/ONNX/etc., installed directly on the Jetson
once it's up) is `JETSON_ENV_SETUP.md`, a separate concern.

**JetPack version: open, not pinned.** Originally targeting 6.2.3 (see `JETSON_ENV_SETUP.md`
for that reasoning), but getting SDK Manager/this account's catalog to recognize 6.2.3
specifically hit repeated friction (GUI defaulted to 7.2.1, CLI queries only surfaced 7.x).
Decision (user, 2026-08-25): stop fighting for 6.2.3 specifically and use whatever JetPack
7.x version the CLI query (step 5 below) actually returns for `JETSON_AGX_ORIN_TARGETS` —
NVIDIA's own release notes confirm 7.2.1 supports the AGX Orin, not just Thor, so this is a
real, valid path, not a downgrade-of-necessity. **Consequence, not yet done**: the
CUDA/onnxruntime-gpu/torch wheel-index research in `JETSON_ENV_SETUP.md` is pinned to
6.2.3's numbers (CUDA 12.6, L4T 36.5.2) and needs redoing against whatever 7.x version this
lands on (confirmed for JetPack 7.2.1 so far: CUDA 13.2, TensorRT 10.16, Python 3.12 wheels
— **and these wheels were found to be prerelease/undocumented, needing pip's `--pre` flag**,
a real, less-mature-ecosystem tradeoff of going with 7.x over 6.2.x).

**Not yet confirmed end-to-end on real hardware** — captures the procedure and findings
from getting this far, not a verified-working recipe. Update this doc once a flash has
actually completed successfully.

## Host OS requirement

- JetPack 6.2.x wants Ubuntu 22.04 or 20.04 as the SDK Manager host.
- **JetPack 7.2.x wants Ubuntu 24.04** — support for 24.04 as an SDK Manager host was only
  added starting with JetPack 7.2, which also explains the "this release is not officially
  supported on your operating system" warning seen in the GUI when the host was still
  20.04. Since we're on the 7.x path now, **use Ubuntu 24.04 LTS** for the native
  boot/dual-boot — any current point release (24.04, .1, .2, .3...) is fine, they're all
  the same base OS for this purpose, just get it from `releases.ubuntu.com/24.04/` if it's
  not front-and-center on Ubuntu's homepage (which likely highlights whatever LTS is newest
  by the time you're reading this — don't use that one, it won't be validated for JetPack
  yet, going by how recently 24.04 support itself landed).

## Recommended: native Ubuntu (dual-boot or Live USB)

Both VirtualBox and WSL2 got the board to enumerate over USB but failed at the *next*
step — reading back the chip's actual identity over the recovery-mode link (VirtualBox:
explicit `ECID read failed` error; WSL2: `sdkmanager --list-connected` showed the device
present but with a blank Device Name). Native Ubuntu removes the virtualization/passthrough
layer entirely, which is the one variable both failures had in common.

1. **Get online** (WiFi/Ethernet) — needed for the SDK Manager download and the JetPack
   image download later.
2. **Install SDK Manager**, as a local file path (not a bare package-name lookup — see
   "Known issue" below for why that distinction matters):
   ```bash
   cd ~/Downloads
   sudo apt install ./sdkmanager_<version>_amd64.deb
   ```
3. **Connect the Jetson directly** — plain USB-C cable, no passthrough tooling needed.
   Board fully powered off, then into Force Recovery Mode (hold Force Recovery,
   press-and-release Reset while holding it), cable into the correct Type-C port.
4. **Confirm it actually identifies — the real test of whether native fixed the problem**:
   ```bash
   sdkmanager --list-connected Jetson
   ```
   Look for the **Device Name** column to populate (e.g. "Jetson AGX Orin"). If it's still
   blank here, that points at the board/cable itself rather than the virtualization layer,
   which neither of the two earlier failures had ruled out.
5. **Query the real available version for this board** — don't assume a version string,
   confirmed this session that guessing wastes attempts:
   ```bash
   sdkmanager --query non-interactive --product Jetson --target JETSON_AGX_ORIN_TARGETS --show-all-versions
   ```
6. **Flash**, using the version string from step 5:
   ```bash
   sdkmanager --cli --action install \
     --login-type devzone \
     --product Jetson \
     --target-os Linux \
     --version <VERSION_FROM_STEP_5> \
     --target JETSON_AGX_ORIN_TARGETS \
     --flash \
     --license accept \
     --stay-logged-in true
   ```
   Notes on this command, all confirmed against this SDK Manager build's own `--help`
   output (not guessed):
   - `--flash` is a bare presence flag — **not** `--flash all` (an earlier draft of this
     doc had that wrong).
   - No `--host` flag means host-side components (CUDA toolkit, Computer Vision, Developer
     Tools — ~5GB, and this project doesn't use them, since dependencies install directly
     on the Jetson post-boot per `JETSON_ENV_SETUP.md`) are **not** selected by default —
     confirmed via `--help`: *"When this parameter is set, the host components will be
     selected."* No `--select`/`--deselect` needed to achieve an OS-only flash.
   - `--target JETSON_AGX_ORIN_TARGETS` is confirmed correct straight from this build's own
     `--help` Jetson example.
7. **After it finishes**: the board reboots on its own. Flashed headless, so the first real
   check is whether it's reachable on the network (`ping`/SSH once it's had a minute to
   boot) — that's where Track 1 bring-up (`run_jetson_standalone.py`) actually starts.

### Setting up native Ubuntu, if not already done

The same Ubuntu 24.04 USB works for either a quick **Live USB session** ("Try Ubuntu," no
install, everything above happens in one sitting) or a proper **dual-boot install**
("Install Ubuntu" → "Install Ubuntu alongside Windows Boot Manager" when the installer asks
about disk setup) — recommended if this machine will be doing this kind of work repeatedly,
which this project will. Dual-boot notes:
- Back up important data first — standard precaution before touching partitions.
- Shrink the Windows partition first (Disk Management → Shrink Volume) to free space —
  150-250GB+ is reasonable given this is an ML/robotics project that'll accumulate
  models/datasets over time.
- Disable Windows Fast Startup first (Power Options → "Choose what the power buttons do")
  — a common real cause of filesystem issues on dual-boot setups if left on.
- Secure Boot usually doesn't need disabling — modern Ubuntu installers ship a signed
  bootloader that coexists with it.

## Repo location note

The project repo being cloned on whatever Linux environment is being used for flashing
does not put anything on the Jetson itself — once the board is up and networked,
`host_software/` needs to get onto the Jetson separately (`git clone` directly on the
device, or `scp`/`rsync` over the network). Distinct step from flashing.

## Why not WSL2

Tried after VirtualBox (see below), on the reasoning that it's NVIDIA's actual documented
Windows path (unlike VirtualBox). Got further than VirtualBox — USB passthrough via
`usbipd-win` worked, `lsusb` inside WSL2 correctly showed the board (`0955:7023`, NVIDIA
Corp. APX) — but `sdkmanager --list-connected Jetson` showed the device present with a
**blank Device Name**, meaning the deeper chip-identification handshake wasn't completing,
and every version/target query subsequently came back empty. Community guidance found
mid-session states plainly that WSL2 + USB-over-IP "is not a capable development
environment for the Jetson platform" for this kind of bring-up, despite NVIDIA's docs
describing it as supported — matches what was observed here. Not revisited further once
native Ubuntu was chosen instead; the `usbipd-win` install fix below is kept since it's
still generally useful if WSL2 is ever revisited.

### Known issue: `usbipd-win` install fails with exit code 1603

Seen when VirtualBox is also installed on the same Windows machine. **Does not require
uninstalling VirtualBox** — confirmed root cause from the MSI install log:

```
Error 1921. Service 'VirtualBox USB Monitor Service' (VBoxUSBMon) could not be stopped.
```

`usbipd-win`'s installer needs to briefly stop `VBoxUSBMon` during its own driver setup to
avoid the two USB filter drivers conflicting mid-install — it doesn't replace or remove
anything VirtualBox owns permanently.

**Fix, in order:**
1. Reboot Windows, then retry the install as-is. The failing install's log showed
   `MsiSystemRebootPending = 1` (a pending reboot already flagged) — a very common, direct
   cause of exactly this "service could not be stopped" failure, despite the message
   wording suggesting a permissions problem (the log confirms the installer *was* running
   fully elevated).
2. If still failing: close any running VirtualBox VMs first (an active VM using USB
   passthrough holds `VBoxUSBMon` busy).
3. Still stuck: stop the service manually right before installing, from an elevated
   PowerShell — `Stop-Service VBoxUSBMon` — then rerun the installer.

`VBoxUSBMon` restarts normally on its own afterward; VirtualBox keeps working for
whatever else it's used for.

### Also found along the way: `sdkmanager` package-name collision

`sudo apt install sdkmanager` (searching Ubuntu's repos by name, no path) resolves to an
unrelated Google Android SDK command-line-tools package that happens to register the same
virtual package name — not NVIDIA's tool at all. Always install NVIDIA's `.deb` as a local
file path instead:
```bash
sudo apt install ./sdkmanager_<version>_amd64.deb
```

## Why not VirtualBox

Tried first, abandoned after two distinct real failures:
1. `ECID read failed` during the actual device flash — a well-documented VirtualBox
   failure class: the recovery-mode USB device disconnects/re-enumerates mid-flash as
   part of the normal handshake, and VirtualBox's USB device filter frequently fails to
   automatically re-capture it, so the flash loses the connection at exactly the wrong
   moment. NVIDIA doesn't officially support flashing via VirtualBox at all.
2. An `apt-get update` repository-access failure on the VM's host-side component install —
   root cause: the VM was running Ubuntu 20.04, which is past standard EOL (April 2025),
   so its default `sources.list` pointed at mirrors that no longer serve that release.
