# Jetson VNC Debug Access — Start/Stop Runbook

Procedure for viewing `run_jetson_standalone.py`'s `cv2.imshow` debug window (Track 1,
non-`--headless` mode) from a Windows laptop over the Jetson's SSH connection. Covers
the standard plain-SSH session (the baseline connection for everything else) plus the
VNC-specific steps layered on top of it. Assumes the one-time setup below has already
been done on the Jetson; this doc is the per-session start/stop steps, not a first-time
tutorial.

## Standard SSH session — start/stop

The baseline connection everything else in this doc builds on top of.

**Start** — from a Windows PowerShell window:

```powershell
ssh jetson-orin@192.168.55.1
```

`192.168.55.1` is the Jetson's default USB device-mode network address (see the
bandwidth note under Troubleshooting); use the real IP if you've moved to the RJ45
Ethernet port instead. On the very first connection from a given Windows machine, `ssh`
will ask you to confirm the host key fingerprint — accept it (`yes`); this only happens
once per client per Jetson. Enter the `jetson-orin` account password when prompted.

**Stop (end the session, leave the board running)** — the normal case between debug
runs:

```bash
exit
```

(or `Ctrl+D`). This just closes your shell; the Jetson keeps running, and any
`vncserver :1` session you started stays up independently — safe to reconnect later
without redoing anything.

**Stop (power off the board)** — only when you're actually done with the hardware for
the day, e.g. before unplugging power. Don't just cut power at the barrel jack —
that risks corrupting the eMMC/NVMe filesystem mid-write:

```bash
sudo shutdown -h now
```

Wait for the board's fan to stop and status LEDs to go dark before disconnecting power.
Use `sudo reboot` instead if you want to restart rather than power off.

## One-time setup (already done — reference only)

- `sudo apt install -y tigervnc-standalone-server tigervnc-common openbox`
- `vncpasswd` run once (sets `~/.vnc/passwd`, independent of the Linux login password)
- `~/.vnc/xstartup` contains:

  ```sh
  #!/bin/sh
  exec openbox-session
  ```

  (`exec`, not `openbox-session &` — backgrounding it makes the wrapper think the
  session exited immediately and it kills the display; see Troubleshooting.)

If any of the above is missing on a fresh Jetson flash, redo it before following the
steps below.

## Start

**1. Jetson side — start the virtual VNC display** (over an existing SSH session):

```bash
vncserver :1 -geometry 1280x800 -localhost yes
```

Confirm it's up:

```bash
vncserver -list
```

**2. Windows side — open the SSH tunnel** (PowerShell; leave this window open for the
whole debug session):

```powershell
ssh -L 5901:localhost:5901 jetson-orin@192.168.55.1
```

`192.168.55.1` is the Jetson's default USB device-mode network address (USB-C cable
presenting as a virtual Ethernet adapter) — confirm with `hostname -I` on the Jetson if
this has changed (e.g. after moving to the real RJ45 Ethernet port).

**3. Windows side — connect the VNC viewer** (TigerVNC Viewer):

```text
localhost:1
```

Enter the VNC password from `vncpasswd`. A **black background is expected** — bare
Openbox has no wallpaper/panel by default; that's normal, not a failure.

**4. Jetson side — run the pipeline with GUI enabled**, targeting the virtual display
(new SSH session, or the tunnel window if you didn't background it with `-N`):

```bash
DISPLAY=:1 python3 host_software/ml_jetson_vla/runtime/run_jetson_standalone.py
```

Omit `--headless` — that flag is what disables the `cv2.imshow` call this whole setup
exists to view. The debug window appears inside the VNC view from step 3.

## Stop

1. Stop the pipeline script (`Ctrl+C` in the SSH session running it).
2. Kill the virtual display on the Jetson:

   ```bash
   vncserver -kill :1
   ```

   This terminates the `Xvnc` process for display `:1` and removes its PID/lock files
   (`~/.vnc/<hostname>:1.pid`, the socket) — it's the actual shutdown of the server, not
   just disconnecting from it. Run `vncserver -list` afterward to confirm `:1` no
   longer appears.
3. Close the Windows PowerShell tunnel window (or kill the backgrounded `ssh.exe`
   process if you started it with `-N -f`). Closing the VNC viewer alone does **not**
   stop the server or tunnel — both keep running until explicitly killed.

**Order matters conceptually, not mechanically**: closing only the *viewer* window
(step 3's client) leaves `vncserver :1` running on the Jetson — harmless, and actually
convenient if you plan to reconnect later without redoing the setup. Closing only the
*server* (step 2) without closing the viewer just leaves the viewer showing a frozen
last frame until you close it manually. To fully tear the debug session down (e.g.
before powering off the Jetson via the Standard SSH section above), do both: kill the
server with `vncserver -kill :1`, then close the viewer/tunnel on the Windows side.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `vncserver` log: "Session startup ... cleanly exited too early (< 3 seconds)" | `xstartup` backgrounds the window manager (`&`) instead of `exec`-ing it. Fix the file as shown in One-time setup, then retry `vncserver :1 ...`. |
| VNC viewer shows a black screen, nothing else | Expected — bare Openbox has no desktop background/panel. Verify the display is actually live with `DISPLAY=:1 xclock` (needs `x11-apps`) from an SSH session; a window appearing confirms the chain works. |
| VNC feels laggy / low framerate | You're likely on the USB device-mode link (`192.168.55.1`), not physical Gigabit Ethernet — bandwidth is capped by the USB link, not anything VNC-side. Move the cable to the Jetson's RJ45 port for a real Ethernet link, or lower `-depth`/viewer compression settings to reduce data per frame. Changing the VNC port number does **not** affect bandwidth. |
| Viewer connects then instantly disconnects | VNC password mismatch — rerun `vncpasswd` on the Jetson. |
| `ssh -L ...` tunnel window closed by accident | VNC viewer will freeze/disconnect — reopen the tunnel command and reconnect the viewer; the Jetson-side `vncserver :1` session itself is unaffected and does not need restarting. |
