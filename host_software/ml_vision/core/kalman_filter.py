"""Constant-velocity Kalman filter for 2D ball position tracking.

Meant as a principled replacement for PredictionGate's EMA + dead-band
smoothing (main_onnx_shared_vision_audio.py): predicting forward with an
explicit velocity term should track a moving ball with less lag than an EMA
at the same noise-suppression level, since the EMA only ever averages the
past. Jump-gating / no-ball / seed-phase detection in PredictionGate are
independent of the smoothing method and are left untouched.

R and Q are NOT guessed here -- they're meant to come from
estimate_kalman_noise_params.py run against a real --scripted-sequence
recording (raw CNN vs. touch ground truth). The defaults below are only a
fallback so --kalman can be smoke-tested before real parameters exist; treat
any result obtained with the defaults as illustrative, not trustworthy.
"""

import numpy as np


class KalmanFilter2D:
    """State: [x, y, vx, vy] (mm, mm/s). Measurement: [x, y] (mm)."""

    H = np.array([[1.0, 0.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, 0.0]])

    def __init__(self, r: np.ndarray = None, q: np.ndarray = None, initial_p_diag=(25.0, 25.0, 2500.0, 2500.0)):
        # Fallback R/Q (illustrative only -- see module docstring). ~5mm std
        # measurement noise, generous process noise so an unvalidated filter
        # leans on measurements rather than an unproven motion model.
        self.R = np.asarray(r, dtype=np.float64) if r is not None else np.eye(2) * 25.0
        self.Q = np.asarray(q, dtype=np.float64) if q is not None else np.diag([50.0, 50.0, 5000.0, 5000.0])
        self._initial_p_diag = np.asarray(initial_p_diag, dtype=np.float64)
        self.x = np.zeros(4, dtype=np.float64)
        self.P = np.diag(self._initial_p_diag)
        self._initialized = False

    def reset(self, x_mm: float, y_mm: float, vx: float = 0.0, vy: float = 0.0):
        """Re-seed the filter at a known position (e.g. on ball placement or
        after a stall) instead of letting it converge from a cold P."""
        self.x = np.array([x_mm, y_mm, vx, vy], dtype=np.float64)
        self.P = np.diag(self._initial_p_diag)
        self._initialized = True

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def position(self):
        return float(self.x[0]), float(self.x[1])

    @property
    def velocity(self):
        return float(self.x[2]), float(self.x[3])

    def predict(self, dt_s: float):
        if not self._initialized:
            return
        f = np.array([[1.0, 0.0, dt_s, 0.0],
                      [0.0, 1.0, 0.0, dt_s],
                      [0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0]])
        self.x = f @ self.x
        self.P = f @ self.P @ f.T + self.Q

    def update(self, x_mm: float, y_mm: float):
        """Returns the post-update (x, y) position estimate."""
        if not self._initialized:
            self.reset(x_mm, y_mm)
            return self.position
        z = np.array([x_mm, y_mm], dtype=np.float64)
        y = z - self.H @ self.x
        s = self.H @ self.P @ self.H.T + self.R
        k = self.P @ self.H.T @ np.linalg.inv(s)
        self.x = self.x + k @ y
        self.P = (np.eye(4) - k @ self.H) @ self.P
        return self.position

    def predict_only_position(self):
        """Position estimate without folding in a new measurement -- used to
        hold a sensible value across a jump-gate-rejected or no-ball frame
        instead of freezing at the last transmitted point."""
        return self.position

    def deinitialize(self):
        """Drop to the cold, unseeded state -- mirrors PredictionGate's
        reset_smoothing()/ball-lost paths, where the next accepted
        measurement should re-seed rather than fold into stale state."""
        self._initialized = False
