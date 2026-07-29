from collections import deque


class TargetStateMachine:
    def __init__(self, history_size=10):
        self.current_target_name = "center"
        self.valid_targets = [
            "center", "hold",
            "blue", "green", "red", "yellow",
            "grey", "black", "cyan", "purple", "orange", "pink", "brown"
        ]
        self.hold_x = 0.0
        self.hold_y = 0.0

        self.history_size = history_size
        self.marker_history = {
            "blue": deque(maxlen=history_size),
            "green": deque(maxlen=history_size),
            "red": deque(maxlen=history_size),
            "yellow": deque(maxlen=history_size),
        }

        self.auto_hold_tolerance_mm = 8.0
        self.auto_hold_required_frames = 6
        self._on_target_frames = 0

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
        if command in ("hold", "stop"):
            self.current_target_name = "hold"
            self.hold_x = float(cam_x)
            self.hold_y = float(cam_y)
            self._on_target_frames = 0
            print(f"[{command.upper()}] Holding at ({self.hold_x:.1f}, {self.hold_y:.1f})")

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

            step = 5.0  # mm, applied once per command edge
            if command == "forward":
                self.hold_y += step
            elif command == "backward":
                self.hold_y -= step
            elif command == "left":
                self.hold_x -= step
            elif command == "right":
                self.hold_x += step

            # Clamp to platform bounds.
            self.hold_x = max(-70.0, min(70.0, self.hold_x))
            self.hold_y = max(-55.0, min(55.0, self.hold_y))
            self._on_target_frames = 0
            print(f"[{command.upper()}] Nudged target to ({self.hold_x:.1f}, {self.hold_y:.1f})")

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
            print(f"[AUTO HOLD] Locked at ({self.hold_x:.1f}, {self.hold_y:.1f}) after reaching target.")

    # -----------------------------------------------------------------
    # Marker history / target resolution
    # -----------------------------------------------------------------
    def update_markers(self, marker_coords):
        for name, coords in marker_coords.items():
            if name in self.marker_history:
                self.marker_history[name].append(coords)

    def get_target_coords(self, marker_coords=None):
        # Accept optional live marker coords for call-site compatibility.
        if marker_coords:
            self.update_markers(marker_coords)

        if self.current_target_name == "center":
            return 0.0, 0.0
        if self.current_target_name == "hold":
            return self.hold_x, self.hold_y

        # Target is a color. Use averaged history if we have any.
        history = self.marker_history.get(self.current_target_name)
        if history and len(history) > 0:
            avg_x = sum(pt[0] for pt in history) / len(history)
            avg_y = sum(pt[1] for pt in history) / len(history)
            return avg_x, avg_y

        # No history for the target -> fall back to center.
        return 0.0, 0.0

