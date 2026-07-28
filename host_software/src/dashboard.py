"""
src/dashboard.py
----------------
Real-time operator dashboard for the ball-balance system.

Architecture
------------
Runs in a SEPARATE PROCESS (not thread) so the main inference loop
has ZERO GIL contention and ZERO Tk cross-thread synchronization.

update() calls queue.put_nowait() which is a non-blocking pipe write
returning in microseconds. If the dashboard subprocess is behind it
silently drops the frame -- the main loop is never stalled.

Public API
----------
    dashboard = LiveDashboard()
    dashboard.start()
    dashboard.update(cam_x, cam_y, target_x, target_y,
                     target_name, marker_coords, command, fps, total_ms)
    dashboard.stop()
"""

import multiprocessing as mp
import time
from datetime import datetime


# ---------------------------------------------------------------------------
# Child-process entry point (must be a top-level function for mp spawn on Win)
# ---------------------------------------------------------------------------

def _dashboard_worker(queue):
    """Runs in the child process. Owns the Matplotlib window."""
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    PLAT_X, PLAT_Y = 70.0, 55.0

    CMD_COLORS = {
        "hold": "#F5A623", "stop": "#F5A623",
        "center": "#9B9B9B",
        "forward": "#F8E71C", "backward": "#F8E71C",
        "left": "#F8E71C", "right": "#F8E71C",
    }
    GO_COLOR = "#7ED321"
    DEFAULT_COLOR = "#AAAAAA"
    MARKER_COLORS = {
        "blue": "#4A90E2", "green": "#7ED321", "red": "#E74C3C",
        "yellow": "#F1C40F", "grey": "#95A5A6", "black": "#7F8C8D",
        "cyan": "#1ABC9C", "purple": "#9B59B6", "orange": "#E67E22",
        "pink": "#FF6B9D", "brown": "#A04000",
    }

    # --- Build figure ---
    fig = plt.figure(figsize=(11, 7), facecolor="#1a1a2e")
    try:
        fig.canvas.manager.set_window_title("Ball Balance -- Live Dashboard")
    except Exception:
        pass

    gs = fig.add_gridspec(
        2, 2, height_ratios=[3, 1], width_ratios=[3, 2],
        hspace=0.35, wspace=0.3,
        left=0.07, right=0.97, top=0.90, bottom=0.05,
    )
    ax_plot = fig.add_subplot(gs[0, 0])
    ax_cmd  = fig.add_subplot(gs[0, 1])
    ax_log  = fig.add_subplot(gs[1, :])

    for ax in (ax_plot, ax_cmd, ax_log):
        ax.set_facecolor("#16213e")
        for spine in ax.spines.values():
            spine.set_edgecolor("#0f3460")

    fig.text(0.5, 0.96, "Ball Balance  |  Live Dashboard",
             ha="center", va="center",
             fontsize=13, color="#e0e0e0", fontweight="bold")
    fps_txt = fig.text(0.97, 0.96, "FPS: --  |  Latency: -- ms",
                       ha="right", va="center", fontsize=9, color="#888888")

    plt.show(block=False)
    plt.pause(0.05)

    def init_plot(ax):
        ax.set_xlim(-PLAT_X - 5, PLAT_X + 5)
        ax.set_ylim(-PLAT_Y - 5, PLAT_Y + 5)
        ax.set_aspect("equal")
        ax.set_title("Position (mm)", color="#cccccc", fontsize=10, pad=4)
        ax.tick_params(colors="#555555", labelsize=7)
        ax.set_xlabel("X (mm)", color="#555555", fontsize=8)
        ax.set_ylabel("Y (mm)", color="#555555", fontsize=8)
        ax.add_patch(patches.Rectangle(
            (-PLAT_X, -PLAT_Y), 2 * PLAT_X, 2 * PLAT_Y,
            linewidth=1.2, edgecolor="#0f3460", facecolor="none", linestyle="--",
        ))
        ax.axhline(0, color="#2a2a4a", linewidth=0.6)
        ax.axvline(0, color="#2a2a4a", linewidth=0.6)

    state = dict(
        cam_x=0.0, cam_y=0.0,
        target_x=0.0, target_y=0.0,
        target_name="center",
        marker_coords={},
        command=None,
        fps=0.0, total_ms=0.0,
    )
    cmd_history = []   # [(ts_str, cmd_str), ...]
    period = 0.10      # 10 Hz redraw
    last_draw = 0.0

    while True:
        # Drain queue -- keep only the latest item
        latest = None
        try:
            while True:
                latest = queue.get_nowait()
        except Exception:
            pass

        if latest == "STOP":
            break

        if latest is not None:
            cmd = latest.get("command")
            if cmd is not None and cmd != state.get("command"):
                ts = datetime.now().strftime("%H:%M:%S")
                cmd_history.insert(0, (ts, cmd))
                del cmd_history[10:]
            state.update(latest)

        now = time.perf_counter()
        if now - last_draw < period:
            plt.pause(0.01)
            continue
        last_draw = now

        # --- Position plot ---
        ax_plot.cla()
        init_plot(ax_plot)
        for name, pos in state["marker_coords"].items():
            mx, my = pos
            color = MARKER_COLORS.get(name, "#ffffff")
            ax_plot.plot(mx, my, "s", color=color, markersize=8, alpha=0.85)
            ax_plot.text(mx, my + 4, name,
                         color=color, fontsize=6, ha="center", va="bottom")
        ax_plot.plot(state["target_x"], state["target_y"], "+",
                     color="#E74C3C", markersize=18, markeredgewidth=2,
                     label="Target: " + state["target_name"])
        ax_plot.plot(state["cam_x"], state["cam_y"], "o",
                     color="#4A90E2", markersize=10,
                     markeredgecolor="#ffffff", markeredgewidth=0.8,
                     label="Ball")
        ax_plot.legend(loc="upper right", fontsize=7,
                       facecolor="#0f3460", edgecolor="#0f3460",
                       labelcolor="#cccccc", markerscale=0.8)

        # --- Command panel ---
        ax_cmd.cla()
        ax_cmd.set_xlim(0, 1)
        ax_cmd.set_ylim(0, 1)
        ax_cmd.axis("off")
        ax_cmd.set_title("Audio Command", color="#cccccc", fontsize=10, pad=4)
        raw_cmd = state["command"] or "--"
        if raw_cmd.startswith("go_"):
            cmd_color   = GO_COLOR
            display_cmd = "GO " + raw_cmd.split("_", 1)[1].upper()
        else:
            cmd_color   = CMD_COLORS.get(raw_cmd, DEFAULT_COLOR)
            display_cmd = raw_cmd.upper()
        ax_cmd.text(0.5, 0.72, display_cmd, ha="center", va="center",
                    fontsize=20, fontweight="bold", color=cmd_color,
                    transform=ax_cmd.transAxes)
        ax_cmd.text(0.5, 0.48, "Active target:", ha="center", va="center",
                    fontsize=8, color="#555555", transform=ax_cmd.transAxes)
        ax_cmd.text(0.5, 0.33, state["target_name"].upper(),
                    ha="center", va="center", fontsize=14, fontweight="bold",
                    color="#e0e0e0", transform=ax_cmd.transAxes)
        ax_cmd.text(0.5, 0.12,
                    "Ball  ({:+.1f}, {:+.1f}) mm".format(state["cam_x"], state["cam_y"]),
                    ha="center", va="center", fontsize=8, color="#666666",
                    transform=ax_cmd.transAxes)
        ax_cmd.text(0.5, 0.03,
                    "Target ({:+.1f}, {:+.1f}) mm".format(state["target_x"], state["target_y"]),
                    ha="center", va="center", fontsize=8, color="#666666",
                    transform=ax_cmd.transAxes)

        # --- Command log ---
        ax_log.cla()
        ax_log.set_xlim(0, 1)
        ax_log.set_ylim(0, 1)
        ax_log.axis("off")
        ax_log.set_title("Recent Commands", color="#cccccc", fontsize=9, pad=3)
        if not cmd_history:
            ax_log.text(0.5, 0.5, "No commands received yet",
                        ha="center", va="center", fontsize=8, color="#444444",
                        transform=ax_log.transAxes)
        else:
            col_width = 0.5
            for i, (ts, cmd) in enumerate(cmd_history):
                col = i % 2
                row = i // 2
                x = col * col_width + 0.02
                y = 0.88 - row * 0.22
                if y < 0:
                    break
                ax_log.text(x, y, "[{}]  {}".format(ts, cmd),
                            ha="left", va="top", fontsize=8,
                            color="#cccccc" if i == 0 else "#555555",
                            transform=ax_log.transAxes)

        fps_txt.set_text(
            "FPS: {:.1f}  |  Latency: {:.1f} ms".format(state["fps"], state["total_ms"])
        )
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

    plt.close("all")


# ---------------------------------------------------------------------------
# Public class used by main.py
# ---------------------------------------------------------------------------

class LiveDashboard:
    """
    Spawns the dashboard in a child process.
    update() is a non-blocking put_nowait() -- never stalls the main loop.
    """

    def __init__(self):
        self._queue = None
        self._proc  = None

    def start(self):
        """Spawn the dashboard subprocess."""
        self._queue = mp.Queue(maxsize=4)  # small buffer; old frames dropped
        self._proc  = mp.Process(
            target=_dashboard_worker,
            args=(self._queue,),
            daemon=True,
            name="LiveDashboard",
        )
        self._proc.start()

    def stop(self):
        """Signal subprocess to exit cleanly and wait."""
        if self._queue is not None:
            try:
                self._queue.put_nowait("STOP")
            except Exception:
                pass
        if self._proc is not None:
            self._proc.join(timeout=3.0)
            if self._proc.is_alive():
                self._proc.terminate()

    def update(self, cam_x, cam_y, target_x, target_y,
               target_name, marker_coords, command, fps, total_ms):
        """
        Fire-and-forget update. Puts state into the queue non-blocking.
        Silently drops the frame if the subprocess has not caught up.
        """
        if self._queue is None:
            return
        payload = {
            "cam_x":         float(cam_x),
            "cam_y":         float(cam_y),
            "target_x":      float(target_x),
            "target_y":      float(target_y),
            "target_name":   str(target_name),
            "marker_coords": {k: (float(v[0]), float(v[1]))
                              for k, v in marker_coords.items()},
            "command":       command,
            "fps":           float(fps),
            "total_ms":      float(total_ms),
        }
        try:
            self._queue.put_nowait(payload)
        except Exception:
            pass  # queue full -- silently drop, dashboard will catch up
