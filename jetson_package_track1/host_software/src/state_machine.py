import math
from collections import deque

# Physical platform dimensions (mm) — used to derive nudge step and clamp bounds.
# Source of truth: hardware/platform_templates/ground_truth_manifest.json
_PLATFORM_W = 187.5
_PLATFORM_H = 142.0


class _MarkerPositionFilter:
    """Per-color jump-gate/seed filter for a marker-derived state-machine target.

    See docs/PROJECT_LOGBOOK.md 15/09/2026 ("Marker Target Jump-Gate") for the
    full before/after validation. Design summary -- and why this is NOT a copy
    of main_onnx_shared_vision_audio.py's PredictionGate:

    PredictionGate tracks the BALL, which genuinely moves, so it needs a motion
    model (EMA or Kalman), a jump-gate anchored on that model's prediction, and
    an AWAITING_BALL/"lost -> reseed" escape valve for when the ball is flung
    off-platform and reappears somewhere new. A color marker is a static
    physical object glued to the printed sheet for the life of one session --
    it never legitimately moves, so there is no motion to model and no sense
    in which it can be "lost and reappear elsewhere". The right model here is
    simpler: the current smoothed estimate IS the best guess of a fixed true
    value; a new candidate far from it is noise (most often a different,
    nearby marker's blob getting misclassified as this color -- see the
    marker_classifier.py color-flicker bugs logged the same day), reject it
    outright, and let more *accepted* samples only sharpen the estimate.

    Consequences of that model, each deliberate:
      - No velocity/Kalman term at all.
      - No "AWAITING"/lost phase and no reseed-on-sustained-rejection escape
        valve. A burst of rejected candidates does NOT mean the marker moved,
        so nothing resets -- the anchor simply holds through the whole burst,
        which is exactly what stops a multi-frame misclassification burst from
        ever entering the average (see the logbook entry for the mechanism:
        the old plain rolling mean had no defense once a burst outlasted its
        10-sample window; this filter's anchor never moves during a rejected
        burst, so every sample in it is rejected, not just the first one).
      - A color going undetected for a while (occlusion, ball sitting on it,
        a transient classifier miss) is simply not fed an update that frame --
        not a jump, not evidence of a real move, not gated or decayed. The
        estimate just holds at its last accepted value until a new detection
        arrives, gated the same as any other update.
      - Seeding (seed_window/seed_consistency_mm) exists for the same reason
        PredictionGate seeds the ball: the first few detections of a color
        this session could themselves be noisy before settling, so an initial
        run of mutually-consistent detections is required before the estimate
        is trusted at all. Unlike the ball, there is no camera-frame marker
        exclusion zone check here -- markers are the thing being seeded, not
        an obstacle to avoid seeding on.
    """

    def __init__(
        self,
        jump_threshold_mm: float = 15.0,
        seed_window: int = 5,
        seed_consistency_mm: float = 10.0,
        history_size: int = 30,
    ) -> None:
        # jump_threshold_mm sits between normal same-marker detection jitter
        # (a few mm, comparable to the ball's own ~5.6mm mean error) and real
        # inter-marker spacing on the printed sheet (30mm center-to-center on
        # aruco_markers_03 -- see ground_truth_manifest-derived touch_mm
        # positions in the logbook entry) so a same-marker reading is accepted
        # and a different-marker misclassification is rejected.
        self.jump_threshold_mm = jump_threshold_mm
        self.seed_window = seed_window
        self.seed_consistency_mm = seed_consistency_mm
        self.history_size = history_size

        self._seed_buffer: list = []
        self._history: deque = deque(maxlen=history_size)
        self._estimate = None  # (x, y) tuple once seeded, else None

    @property
    def position(self):
        """Current best-guess (x, y) mm, or None if never yet seeded."""
        return self._estimate

    def update(self, x: float, y: float) -> None:
        """Feed one new raw per-frame detection for this color. Call this only
        on frames where the classifier actually reported this color this
        frame -- for an undetected frame, simply don't call it at all (see
        class docstring: absence is not evidence of a move, so it must not
        gate, reset, or decay anything)."""
        candidate = (float(x), float(y))

        if self._estimate is None:
            # Not yet confirmed: accumulate a sliding seed window and only
            # lock in an estimate once seed_window consecutive detections
            # mutually agree within seed_consistency_mm -- same rationale as
            # PredictionGate's seed phase, scaled for a static target.
            self._seed_buffer.append(candidate)
            if len(self._seed_buffer) > self.seed_window:
                self._seed_buffer.pop(0)
            if len(self._seed_buffer) == self.seed_window:
                cx = sum(p[0] for p in self._seed_buffer) / self.seed_window
                cy = sum(p[1] for p in self._seed_buffer) / self.seed_window
                max_dist = max(
                    math.hypot(p[0] - cx, p[1] - cy) for p in self._seed_buffer
                )
                if max_dist <= self.seed_consistency_mm:
                    self._history.clear()
                    self._history.extend(self._seed_buffer)
                    self._estimate = (cx, cy)
                    self._seed_buffer.clear()
            return

        # Confirmed: jump-gate against the current estimate. Anything beyond
        # jump_threshold_mm is treated as noise and simply discarded -- the
        # estimate is left exactly as it was (no counter, no partial blend,
        # no reset), so a sustained burst of bad candidates is rejected in
        # full, not just its first frame.
        ex, ey = self._estimate
        if math.hypot(candidate[0] - ex, candidate[1] - ey) > self.jump_threshold_mm:
            return

        self._history.append(candidate)
        n = len(self._history)
        hx = sum(p[0] for p in self._history) / n
        hy = sum(p[1] for p in self._history) / n
        self._estimate = (hx, hy)

# Nudge step: 15% of each axis's full range.
# X axis: 0.15 × 187.5 mm ≈ 28 mm per command
# Y axis: 0.15 × 142.0 mm ≈ 21 mm per command
_NUDGE_X = 0.15 * _PLATFORM_W   # ~28.1 mm
_NUDGE_Y = 0.15 * _PLATFORM_H   # ~21.3 mm

# Clamp: 10% inset from the platform edge (90% of each half-range in centred coords).
# X: ±(0.90 × 93.75) = ±84.4 mm   Y: ±(0.90 × 71.0) = ±63.9 mm
_CLAMP_X = 0.90 * (_PLATFORM_W / 2.0)   # ~84.4 mm
_CLAMP_Y = 0.90 * (_PLATFORM_H / 2.0)   # ~63.9 mm


class TargetStateMachine:
    def __init__(self, history_size=10):
        self.current_target_name = "center"
        self.valid_targets = [
            "center",
            "hold",
            "blue",
            "green",
            "red",
            "yellow",
            "grey",
            "black",
            "cyan",
            "purple",
            "orange",
            "pink",
            "brown",
        ]
        self.hold_x = 0.0
        self.hold_y = 0.0

        # history_size here means something different from the old plain
        # rolling-mean deque: it's now the size of the ACCEPTED-only (post
        # jump-gate) sample window each color's _MarkerPositionFilter
        # averages over. Kept as a constructor kwarg for call-site
        # compatibility (no existing caller passes it explicitly).
        self.history_size = history_size
        self.marker_filters = {
            "blue": _MarkerPositionFilter(history_size=history_size),
            "green": _MarkerPositionFilter(history_size=history_size),
            "red": _MarkerPositionFilter(history_size=history_size),
            "yellow": _MarkerPositionFilter(history_size=history_size),
            "black": _MarkerPositionFilter(history_size=history_size),
        }

        self.auto_hold_tolerance_mm = 8.0
        self.auto_hold_required_frames = 6
        self._on_target_frames = 0

        # Ball-loss recovery: on_ball_lost() forces the target to center;
        # maybe_resume_previous_target() switches back to whatever we were
        # pursuing once the ball has genuinely dwelt near center again (same
        # tolerance/dwell pattern as auto-hold), not the instant it's
        # re-acquired.
        self._pre_loss_target = None
        self.recenter_tolerance_mm = 8.0
        self.recenter_required_frames = 6
        self._recenter_frames = 0

        # Edge-trigger state: the last command we actually *acted on*.
        # process_command() is called every loop with a latched command,
        # but we only run the handler when the command changes.
        self._last_command = None

    # -----------------------------------------------------------------
    # Command handling (edge-triggered)
    # -----------------------------------------------------------------
    def process_command(self, command, cam_x=0.0, cam_y=0.0):
        """Called every loop. Only fires on a *change* of command so that
        held audio latches don't re-capture the setpoint or re-nudge each
        frame. Nudges apply exactly once per new command."""
        if command is None:
            return

        # No edge -> nothing to do. This is the whole point of the rewrite.
        if command == self._last_command:
            return
        self._last_command = command

        self._apply_command(command, cam_x, cam_y)

    def _apply_command(self, command, cam_x, cam_y):
        # A fresh explicit command always wins over a queued post-loss
        # resume -- don't snap back to a stale target once the user has
        # asked for something new.
        self._pre_loss_target = None

        if command in ("hold", "stop"):
            self.current_target_name = "hold"
            self.hold_x = float(cam_x)
            self.hold_y = float(cam_y)
            self._on_target_frames = 0
            print(
                f"[{command.upper()}] Holding at ({self.hold_x:.1f}, {self.hold_y:.1f})"
            )

        elif command.startswith("go_"):
            color = command.split("_", 1)[1]
            if color in self.valid_targets:
                self.current_target_name = color
                self._on_target_frames = 0
                print(f"[GO {color.upper()}] Switching target to {color} marker!")

        elif command in ("forward", "backward", "left", "right"):
            # A directional nudge implies we're now holding a fixed point.
            # If we weren't already holding, seed the hold point from the
            # ball's current position, then apply a single nudge.
            if self.current_target_name != "hold":
                self.current_target_name = "hold"
                self.hold_x = float(cam_x)
                self.hold_y = float(cam_y)

            # Camera is mounted with a 180° rotation relative to the platform:
            # physical top-left corner (marker 0) appears at camera bottom-right.
            # Consequence: camera X and physical X are INVERTED.
            #   camera-left  → physical +X  →  hold_x += nudge
            #   camera-right → physical -X  →  hold_x -= nudge
            # Y axis is NOT inverted for these commands:
            #   camera-up (forward)   → physical +Y → hold_y += nudge
            #   camera-down (backward)→ physical -Y → hold_y -= nudge
            if command == "forward":
                self.hold_y += _NUDGE_Y
            elif command == "backward":
                self.hold_y -= _NUDGE_Y
            elif command == "left":
                self.hold_x += _NUDGE_X   # ← camera-left = physical +X
            elif command == "right":
                self.hold_x -= _NUDGE_X   # ← camera-right = physical -X

            # Clamp to 90% of platform half-range (10% inset from edge).
            self.hold_x = max(-_CLAMP_X, min(_CLAMP_X, self.hold_x))
            self.hold_y = max(-_CLAMP_Y, min(_CLAMP_Y, self.hold_y))
            self._on_target_frames = 0
            print(
                f"[{command.upper()}] Nudged target to ({self.hold_x:.1f}, {self.hold_y:.1f})"
            )

        # Unknown commands are ignored (no edge effect).

    # -----------------------------------------------------------------
    # Auto-hold: latch to a fixed point once the ball reaches a marker
    # -----------------------------------------------------------------
    def maybe_auto_hold(self, cam_x, cam_y, marker_coords):
        if self.current_target_name in {"center", "hold"}:
            self._on_target_frames = 0
            return

        if self.current_target_name not in marker_coords:
            self._on_target_frames = 0
            return

        target_x, target_y = marker_coords[self.current_target_name]
        dx = float(target_x) - float(cam_x)
        dy = float(target_y) - float(cam_y)
        dist = (dx * dx + dy * dy) ** 0.5

        if dist <= self.auto_hold_tolerance_mm:
            self._on_target_frames += 1
        else:
            self._on_target_frames = 0

        if self._on_target_frames >= self.auto_hold_required_frames:
            self.current_target_name = "hold"
            self.hold_x = float(cam_x)
            self.hold_y = float(cam_y)
            self._on_target_frames = 0
            # Sync the edge tracker so the still-latched "go_" audio command
            # doesn't immediately look like a new edge and switch us back.
            self._last_command = "hold"
            print(
                f"[AUTO HOLD] Locked at ({self.hold_x:.1f}, {self.hold_y:.1f}) after reaching target."
            )

    # -----------------------------------------------------------------
    # Ball-loss recovery: force center, then resume the previous target
    # -----------------------------------------------------------------
    def on_ball_lost(self):
        """Call exactly once, on the frame the tracker transitions from
        actively tracking the ball to AWAITING_BALL (ball flew off / got
        lost -- not every routine no-ball frame). Remembers whatever we were
        pursuing and forces an immediate switch to center; the actual
        physical move happens naturally once get_target_coords() is next
        read, same as any other target change."""
        if self.current_target_name != "center":
            self._pre_loss_target = self.current_target_name
        self.current_target_name = "center"
        self._on_target_frames = 0
        self._recenter_frames = 0
        print(
            f"[BALL LOST] Forcing target to center"
            + (f" -- will resume '{self._pre_loss_target}' once re-centered." if self._pre_loss_target else ".")
        )

    def maybe_resume_previous_target(self, cam_x, cam_y):
        """Call every tracking-phase frame. Once the ball has dwelt within
        recenter_tolerance_mm of center for recenter_required_frames
        consecutive frames after a ball-lost event, switches back to
        whatever target was active before the loss. No-op otherwise --
        including if the user issued a fresh command in the meantime,
        since _apply_command() already clears _pre_loss_target then."""
        if self._pre_loss_target is None or self.current_target_name != "center":
            return

        dist = (float(cam_x) ** 2 + float(cam_y) ** 2) ** 0.5
        if dist <= self.recenter_tolerance_mm:
            self._recenter_frames += 1
        else:
            self._recenter_frames = 0

        if self._recenter_frames >= self.recenter_required_frames:
            resumed = self._pre_loss_target
            self._pre_loss_target = None
            self._recenter_frames = 0
            self.current_target_name = resumed
            self._on_target_frames = 0
            print(f"[RECOVERED] Ball re-centered -- resuming target '{resumed}'.")

    # -----------------------------------------------------------------
    # Marker history / target resolution
    # -----------------------------------------------------------------
    def update_markers(self, marker_coords):
        """Feed this frame's classifier-reported per-color positions into each
        color's own _MarkerPositionFilter. A color absent from marker_coords
        this frame is simply skipped -- not appended, not reset, not treated
        as a rejection -- see _MarkerPositionFilter's docstring for why."""
        for name, coords in marker_coords.items():
            filt = self.marker_filters.get(name)
            if filt is not None:
                filt.update(coords[0], coords[1])

    def get_target_coords(self, marker_coords=None):
        # Accept optional live marker coords for call-site compatibility.
        if marker_coords:
            self.update_markers(marker_coords)

        if self.current_target_name == "center":
            return 0.0, 0.0
        if self.current_target_name == "hold":
            return self.hold_x, self.hold_y

        # Target is a color. Use its jump-gated/seeded static-position
        # estimate if one has been confirmed yet.
        filt = self.marker_filters.get(self.current_target_name)
        if filt is not None and filt.position is not None:
            return filt.position

        # Not yet confirmed for this target -> fall back to center.
        return 0.0, 0.0
