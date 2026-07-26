from collections import deque

class TargetStateMachine:
    def __init__(self, history_size=10):
        self.current_target_name = "center"
        self.valid_targets = ["center", "blue", "green", "red", "yellow", "hold"]
        self.hold_x = 0.0
        self.hold_y = 0.0
        self.history_size = history_size
        self.marker_history = {
            "blue": deque(maxlen=history_size),
            "green": deque(maxlen=history_size),
            "red": deque(maxlen=history_size),
            "yellow": deque(maxlen=history_size)
        }
        self.auto_hold_tolerance_mm = 8.0
        self.auto_hold_required_frames = 6
        self._on_target_frames = 0
        
    def process_command(self, command, cam_x=0.0, cam_y=0.0):
        if command is None:
            return
            
        if command == "hold" or command == "stop":
            print(f"[{command.upper()}] Holding at current position ({cam_x:.1f}, {cam_y:.1f})!")
            self.current_target_name = "hold"
            self.hold_x = float(cam_x)
            self.hold_y = float(cam_y)
        elif command.startswith("go_"):
            color = command.split("_")[1]
            if color in self.valid_targets:
                print(f"[GO {color.upper()}] Switching target to {color} marker!")
                self.current_target_name = color
                self._on_target_frames = 0

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
            print(f"[AUTO HOLD] Locked at ({self.hold_x:.1f}, {self.hold_y:.1f}) after reaching target.")
                
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
            
        # Target is a color. Do we have history for it?
        history = self.marker_history.get(self.current_target_name)
        if history and len(history) > 0:
            avg_x = sum(pt[0] for pt in history) / len(history)
            avg_y = sum(pt[1] for pt in history) / len(history)
            return avg_x, avg_y
            
        # If we can't see the target and have no history, default to center.
        return 0.0, 0.0
