"""Scripted stand-in for AudioCommandReceiverONNX / KeyboardCommandReceiver,
for collecting controlled ground-truth telemetry to design a Kalman filter's
noise parameters (see PROJECT_LOGBOOK.md 19/08).

Same public interface (get_latest_command(), stop(), latest_inference_time_ms)
as the other two receivers, so it's a drop-in swap at the call site.

Why this exists, not just "run --dummy-audio and type commands manually":
the schedule needs to be reproducible and needs a genuinely uninterrupted long
stationary hold at the start (checking err_x_mm/err_y_mm against the existing
recordings this session showed the measurement-noise estimate is contaminated
by ball motion -- vision and touch sample on two unsynchronized clocks, so a
moving ball inflates apparent "error" with pure timing-skew, not sensor
noise). A human typing commands live can't guarantee a clean, deliberately
boring settle window the way a fixed schedule can.

Two phases, run back to back automatically:
  1. R (measurement noise) window -- no command needed at all, since
     TargetStateMachine already defaults to "center" with no command issued.
     Just a long, deliberately uneventful settle period.
  2. Q (process noise) window -- a sequence of directional nudges
     (state_machine.py's forward/backward/left/right, each ~15% of the
     platform's full range) sweeping the target through varied motion, each
     held long enough for the ball to actually arrive and settle before the
     next nudge fires. Ends on "hold" so the platform doesn't sit mid-command
     when the schedule finishes.

Run via main_onnx_shared_vision_audio.py with --log-csv set to a fresh path
per phase if you want R and Q analyzed from separate files, or just let it
log to one CSV and split by elapsed time in the analysis script -- the printed
schedule below gives you the exact timestamps either way.
"""

import threading
import time

# (elapsed_seconds_from_start, command). Kept as a plain module-level default
# so it's easy to tweak between runs without touching the class.
DEFAULT_SCHEDULE = [
    # --- Phase 1: R -- no command needed, target already defaults to center.
    # Just elapsed time with nothing issued; the long gap IS the point.
    # (first real command fires at 30.0s)
    (30.0, "forward"),
    (34.0, "left"),
    (38.0, "backward"),
    (42.0, "backward"),
    (46.0, "right"),
    (50.0, "right"),
    (54.0, "forward"),
    (58.0, "forward"),
    (62.0, "left"),
    (66.0, "left"),
    (70.0, "hold"),
]


class ScriptedCommandSequencer:
    def __init__(self, schedule=None) -> None:
        self.schedule = schedule if schedule is not None else DEFAULT_SCHEDULE
        self._pending = []  # commands not yet consumed by get_latest_command()
        self._lock = threading.Lock()
        self.running = True
        self.latest_inference_time_ms = 0.0  # call-site compatibility only

        self._start_t = time.perf_counter()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

        print(f"Scripted command sequencer active -- {len(self.schedule)} commands over "
              f"{self.schedule[-1][0]:.0f}s. Phase 1 (R, no command, target=center) runs "
              f"until t={self.schedule[0][0]:.0f}s, then phase 2 (Q, scripted motion) begins.")

    def _run(self) -> None:
        for t_due, command in self.schedule:
            if not self.running:
                return
            now = time.perf_counter() - self._start_t
            wait = t_due - now
            if wait > 0:
                time.sleep(wait)
            if not self.running:
                return
            elapsed = time.perf_counter() - self._start_t
            print(f"\n[SEQUENCER] t={elapsed:.1f}s -- issuing: {command}\n")
            with self._lock:
                self._pending.append(command)
        print("\n[SEQUENCER] Schedule complete -- no further commands will be issued.\n")

    def get_latest_command(self):
        with self._lock:
            if self._pending:
                return self._pending.pop(0)
        return None

    def stop(self) -> None:
        self.running = False
