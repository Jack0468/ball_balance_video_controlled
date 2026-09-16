"""Scripted stand-in for AudioCommandReceiverONNX / KeyboardCommandReceiver, for
reproducible command-schedule-driven evaluation runs. Holds two independent
schedules used for two different purposes -- pick one via the `schedule=`
constructor argument, the class itself is schedule-agnostic.

Same public interface (get_latest_command(), stop(), latest_inference_time_ms)
as the other two receivers, so it's a drop-in swap at the call site.

Why this exists, not just "run --dummy-audio and type commands manually": a
human typing commands live can't guarantee exact, repeatable timing -- both
schedules below need that for what they're measuring/comparing.

--- DEFAULT_SCHEDULE (Kalman filter noise-parameter tuning, see
PROJECT_LOGBOOK.md 19/08) ---
Two phases, run back to back automatically:
  1. R (measurement noise) window -- no command needed at all, since
     TargetStateMachine already defaults to "center" with no command issued.
     Just a long, deliberately uneventful settle period. Needed because
     checking err_x_mm/err_y_mm against earlier recordings showed the
     measurement-noise estimate is contaminated by ball motion -- vision and
     touch sample on two unsynchronized clocks, so a moving ball inflates
     apparent "error" with pure timing-skew, not sensor noise.
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

--- STANDARD_EVAL_SCHEDULE (docs/EVALUATION_STRATEGY.md's Standardized
Evaluation Sequence, updated 2026-09-15) ---
The cross-arm (PID/Expert-Vision-RL/VLA) comparison protocol: 10 commands, each
held exactly 10s, matching that doc verbatim -- keep the two in sync if either
changes. `go_black` deliberately isn't first (untested cold-start target);
`HOLD` fires mid-transit (FORWARD+LEFT drive the ball toward the top-left, then
a color command starts pulling it back across the board, then HOLD interrupts
that transit) rather than between two static targets, since marker positions
are sheet-specific and this way the schedule doesn't depend on which physical
sheet is mounted.

--- random_schedule() (Track 4 unattended data-collection smoke tests,
added 2026-09-15) ---
Not a fixed schedule like the two above -- a generator function that builds
one, drawing from COMMAND_VOCABULARY (the confirmed 10-command active set) so
sessions can be collected automatically without a human typing/speaking each
command live. See its docstring for the exact shape/guarantees.
"""

import random
import threading
import time

# Confirmed active command vocabulary (docs/EVALUATION_STRATEGY.md's 2026-09-15
# update + state_machine.py's actual _apply_command() handling) -- NOT the same
# as state_machine.py's own broader `valid_targets` list, which still carries
# dropped colors (blue/grey) plus some (cyan/purple) with no marker_filters
# entry at all that would silently fail to track. This is the pool
# random_schedule() draws from by default.
COMMAND_VOCABULARY = [
    "go_green", "go_red", "go_yellow", "go_black",
    "hold", "stop",
    "forward", "backward", "left", "right",
]

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

# Mirrors docs/EVALUATION_STRATEGY.md's "Standardized Evaluation Sequence"
# section exactly -- 10 commands x 10s = 100s total. Update both together.
STANDARD_EVAL_SCHEDULE = [
    (0.0, "go_green"),
    (10.0, "go_yellow"),
    (20.0, "forward"),
    (30.0, "left"),
    (40.0, "go_red"),
    (50.0, "hold"),
    (60.0, "go_black"),
    (70.0, "right"),
    (80.0, "backward"),
    (90.0, "stop"),
]


def random_schedule(total_duration_s=200.0, min_interval_s=3.0, max_interval_s=20.0, seed=None, pool=None):
    """Build a randomized (elapsed_seconds, command) schedule for Track 4
    unattended data-collection smoke tests -- same shape ScriptedCommandSequencer
    already consumes, so no changes needed there.

    Total session duration is the primary constraint, not command count: commands
    fire at a randomized `[min_interval_s, max_interval_s)` gap from each other,
    for up to `total_duration_s` seconds total. Command count is a natural
    consequence of the random gaps, not a fixed input.

    - First command at elapsed_seconds=0.0 (matches STANDARD_EVAL_SCHEDULE's
      immediate-start shape, not DEFAULT_SCHEDULE's settle-window lead-in).
    - Then repeatedly draw `gap = rng.uniform(min_interval_s, max_interval_s)` and
      add it to the running elapsed time; append the next command only if the new
      elapsed time is still `< total_duration_s`. So the last command fires
      somewhere before total_duration_s, never at or past it -- no padding or
      truncating to hit exactly total_duration_s, the variability is the point.
    - No immediate repeat of the previous command: state_machine.py's
      process_command() is edge-triggered (`if command == self._last_command:
      return`), so a repeat would just burn a data slot with a no-op.
    - `seed`, if given, drives a local random.Random instance only -- never
      touches the global random module state -- so a specific problematic
      session can be reproduced later without disturbing anything else.
    """
    pool = pool if pool is not None else COMMAND_VOCABULARY
    rng = random.Random(seed)
    schedule = []
    previous = None
    elapsed = 0.0
    while True:
        choices = [c for c in pool if c != previous] if previous is not None else list(pool)
        command = rng.choice(choices)
        schedule.append((elapsed, command))
        previous = command
        elapsed += rng.uniform(min_interval_s, max_interval_s)
        if elapsed >= total_duration_s:
            break
    return schedule


class ScriptedCommandSequencer:
    """Schedule timing doesn't start at construction -- call begin() once the ball is
    actually confirmed on the platform (see run_jetson_standalone.py/
    main_onnx_shared_vision_audio.py's PredictionGate "seeded" transition). Constructing
    this well before that (e.g. while still waiting for the first camera frame, or before
    the ball has settled) would otherwise burn real schedule time on a state where the
    platform can't do anything meaningful yet -- especially costly for DEFAULT_SCHEDULE's
    long settle window, whose whole point is a clean, deliberately uneventful measurement
    period, not "clean plus however long camera/ball warm-up happened to take."
    begin() is safe to call more than once (idempotent, matches threading.Event.set())."""

    def __init__(self, schedule=None) -> None:
        self.schedule = schedule if schedule is not None else DEFAULT_SCHEDULE
        self._pending = []  # commands not yet consumed by get_latest_command()
        self._lock = threading.Lock()
        self.running = True
        self.latest_inference_time_ms = 0.0  # call-site compatibility only

        self._start_event = threading.Event()
        self._start_t = None  # set once begin() unblocks _run(), not at construction
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

        first_t, first_cmd = self.schedule[0]
        if first_t > 0:
            # DEFAULT_SCHEDULE's shape: a long uneventful settle window before the
            # first command -- worth calling out explicitly since it looks like
            # nothing is happening. STANDARD_EVAL_SCHEDULE has no such gap (first_t=0).
            lead_in = f" First {first_t:.0f}s has no command (settle window), then "
        else:
            lead_in = " First command fires immediately: "
        print(f"Scripted command sequencer armed -- {len(self.schedule)} commands over "
              f"{self.schedule[-1][0]:.0f}s, waiting for begin() (ball confirmed on "
              f"platform) before starting the clock.{lead_in}'{first_cmd}'.")

    def begin(self) -> None:
        """Start the schedule's clock now. Call once, when the caller considers the
        platform ready (e.g. PredictionGate's gate_reason == "seeded"). Idempotent."""
        if not self._start_event.is_set():
            self._start_event.set()
            print("[SEQUENCER] begin() -- schedule clock started.")

    def _run(self) -> None:
        self._start_event.wait()  # blocks here until begin() is called
        if not self.running:
            return
        self._start_t = time.perf_counter()
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
        self._start_event.set()  # unblock _run() if it's still waiting on begin()
