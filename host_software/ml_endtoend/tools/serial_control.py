import time
import serial
import torch
import torch.nn as nn
from models.control_model import ControlNet

class ControllerState:
    """Replicates the stateful velocity filter and observation logic from RLControl.cpp"""
    def __init__(self, alpha=0.35, max_motor_step=98.0):
        self.alpha = alpha
        self.max_motor_step = max_motor_step
        self.prev_obs_pos = [0.0, 0.0]
        self.filt_vel = [0.0, 0.0]
        self.have_prev_pos = False

    def reset(self):
        self.prev_obs_pos = [0.0, 0.0]
        self.filt_vel = [0.0, 0.0]
        self.have_prev_pos = False

    def build_observation(self, x_mm, y_mm, target_x_mm, target_y_mm, actual_steps, dt):
        # Prevent division by zero
        if dt < 0.005:
            dt = 0.005

        # Exponential Moving Average velocity filter
        raw_vx, raw_vy = 0.0, 0.0
        if self.have_prev_pos:
            raw_vx = (x_mm - self.prev_obs_pos[0]) / dt
            raw_vy = (y_mm - self.prev_obs_pos[1]) / dt

        self.filt_vel[0] = self.alpha * raw_vx + (1.0 - self.alpha) * self.filt_vel[0]
        self.filt_vel[1] = self.alpha * raw_vy + (1.0 - self.alpha) * self.filt_vel[1]
        
        self.prev_obs_pos[0] = x_mm
        self.prev_obs_pos[1] = y_mm
        self.have_prev_pos = True

        # 9-element observation vector matching C++ code layout
        obs = [
            x_mm,
            y_mm,
            x_mm - target_x_mm,
            y_mm - target_y_mm,
            self.filt_vel[0],
            self.filt_vel[1],
            actual_steps[0],
            actual_steps[1],
            actual_steps[2]
        ]
        return torch.tensor(obs, dtype=torch.float32).unsqueeze(0) # Batch size 1

    def compute_step_targets(self, model, obs_tensor):
        with torch.no_grad():
            action = model(obs_tensor).squeeze(0).numpy() # shape (3,)
        
        # Action in [-1, 1] -> integer step targets
        target_steps = [round(a * self.max_motor_step) for a in action]
        return target_steps


class SerialController:
    def __init__(self, port="/dev/ttyACM0", baudrate=2000000, weights_pth="control_model.pth"):
        self.model = ControlNet()
        self.model.load_state_dict(torch.load(weights_pth, weights_only=True))
        self.model.eval()

        self.state = ControllerState()
        self.ser = serial.Serial(port, baudrate, timeout=0.1)

        # Last known lagged motor state. Starts level: correct cold-start
        # condition, and what we use until the first echo arrives.
        self.actual_steps = [0.0, 0.0, 0.0]
        self.last_time = time.time()

    def _read_actual_steps(self):
        """Drain the serial input buffer, keep the most recent valid
        'A,B,C' position echo. Returns True if a fresh reading was found.
        Non-blocking: only consumes what's already waiting."""
        updated = False
        while self.ser.in_waiting:
            line = self.ser.readline().decode("ascii", errors="ignore").strip()
            parts = line.split(",")
            if len(parts) == 3:
                try:
                    self.actual_steps = [float(parts[0]), float(parts[1]), float(parts[2])]
                    updated = True
                except ValueError:
                    pass  # skip boot messages like "Motors to zero position"
        return updated

    def step(self, x_mm, y_mm, target_x_mm, target_y_mm):
        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        # 1. Read the lagged motor state left by the PREVIOUS command's echo.
        #    This is the currentPosition() the net trained on. If nothing new
        #    arrived, self.actual_steps keeps its last value.
        self._read_actual_steps()

        # 2. Build obs with that state, run the net, scale to step space.
        obs = self.state.build_observation(
            x_mm, y_mm, target_x_mm, target_y_mm, self.actual_steps, dt
        )
        target_steps = self.state.compute_step_targets(self.model, obs)

        # 3. Send the new target. The firmware will echo its currentPosition()
        #    in reply, which we'll read at the top of the NEXT step().
        cmd = f"{target_steps[0]},{target_steps[1]},{target_steps[2]}\n"
        self.ser.write(cmd.encode("utf-8"))

        return target_steps
    
    def home(self):
        self.state.reset()
        self.ser.write(b"0,0,0\n")


# ---------------------------------------------------------------------------
# Quick Test Loop
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Example instantiation
    controller = SerialController(port="/dev/ttyACM0", weights_pth="control_model.pth")
    print("Serial Controller initialized and model loaded successfully.")