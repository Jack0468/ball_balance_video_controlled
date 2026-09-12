"""
Python port of the RL-trained ball-balancing control net currently embedded on
the STM32 (firmware/stm32_ml_control_and_vision/BallBalancingBot/RLControl.cpp +
weights_best_so_far1.h). This is a port, not a retrain -- same weights, same
forward pass, same stateful velocity filter, faithfully replicated so behavior
matches the firmware exactly rather than approximates it.

Why this needs "reverse engineering" rather than a clean load: there is no
original PyTorch/TF training checkpoint in this repo, and the generator script
referenced in the weights header's own comment (export_weights.py) is missing.
The only surviving artifact is the compiled-in C float arrays. This module
parses those arrays directly out of the firmware's own .h file (not hardcoded
copies of the numbers) so it can never silently drift from whatever the
firmware actually ships, even if the weights get re-exported later.

Architecture: 9 -> 32 -> 32 -> 3, tanh hidden, linear output clipped to [-1,1],
scaled by MAX_MOTOR_STEP and rounded to integer step-space targets.
IMPORTANT: the output is ALREADY in step-space -- it needs no angle conversion
and feeds directly into a stepper's target_position (see AGENTS.md's ControlNet
I/O contract note and agent_fpga.md's Option B discussion). What actually needs
to change to run this on the Jetson is (1) this extraction/port, and (2) the
STM32 firmware needs a new input mode that accepts target_steps directly and
drives to them via MotorControl.cpp, bypassing RLControl.cpp's own on-device
inference -- otherwise the STM32 keeps computing control itself regardless of
what runs here. That firmware change is not yet done; this file is the
Jetson-side half only.

NOT YET HARDWARE-VALIDATED: no STM32/Jetson access in this environment to
compare outputs against the real firmware. Validated here only for structural
correctness (shapes, param count) -- see self_test() below.
"""

import re
from pathlib import Path

import numpy as np

DEFAULT_WEIGHTS_PATH = (
    Path(__file__).resolve().parents[3]
    / "firmware"
    / "stm32_ml_control_and_vision"
    / "BallBalancingBot"
    / "weights_best_so_far1.h"
)

# Mirrors RLControl.cpp exactly -- do not change these without changing the
# firmware too, they must stay identical for behavior to match.
MAX_MOTOR_STEP = 98.0
VEL_FILTER_ALPHA = 0.35
CONTROL_DT = 0.0333  # nominal 30 Hz; real dt is passed in per-call by the caller


def _parse_array(header_text: str, name: str) -> np.ndarray:
    match = re.search(rf"static const float {name}\[\]\s*=\s*\{{(.*?)\}};", header_text, re.S)
    if match is None:
        raise ValueError(f"Could not find array '{name}' in weights header")
    values = [
        float(v.rstrip("fF"))
        for v in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?f?", match.group(1))
    ]
    return np.array(values, dtype=np.float32)


def _parse_dims(header_text: str) -> dict:
    dims = {}
    for name in ("NN_IN", "NN_H1", "NN_H2", "NN_OUT"):
        m = re.search(rf"#define {name} (\d+)", header_text)
        if m is None:
            raise ValueError(f"Could not find dimension macro '{name}' in weights header")
        dims[name] = int(m.group(1))
    return dims


class ControlNet:
    """Stateful port of RLControl.cpp's rl_infer(). One instance per robot --
    the velocity filter and previous-position state are per-instance, matching
    the firmware's static globals (rl_reset_state() -> reset_state())."""

    def __init__(self, weights_path: Path = DEFAULT_WEIGHTS_PATH):
        text = Path(weights_path).read_text()
        dims = _parse_dims(text)
        self.n_in, self.n_h1, self.n_h2, self.n_out = (
            dims["NN_IN"],
            dims["NN_H1"],
            dims["NN_H2"],
            dims["NN_OUT"],
        )

        # Firmware layout is row-major (in_dim, out_dim): W[i * out_dim + j].
        self.w1 = _parse_array(text, "NN_W1").reshape(self.n_in, self.n_h1)
        self.b1 = _parse_array(text, "NN_B1")
        self.w2 = _parse_array(text, "NN_W2").reshape(self.n_h1, self.n_h2)
        self.b2 = _parse_array(text, "NN_B2")
        self.w3 = _parse_array(text, "NN_W3").reshape(self.n_h2, self.n_out)
        self.b3 = _parse_array(text, "NN_B3")

        expected = {
            "NN_W1": self.n_in * self.n_h1,
            "NN_B1": self.n_h1,
            "NN_W2": self.n_h1 * self.n_h2,
            "NN_B2": self.n_h2,
            "NN_W3": self.n_h2 * self.n_out,
            "NN_B3": self.n_out,
        }
        got = {
            "NN_W1": self.w1.size,
            "NN_B1": self.b1.size,
            "NN_W2": self.w2.size,
            "NN_B2": self.b2.size,
            "NN_W3": self.w3.size,
            "NN_B3": self.b3.size,
        }
        if expected != got:
            raise ValueError(f"Weight array size mismatch: expected {expected}, got {got}")

        self.total_params = sum(expected.values())
        self.reset_state()

    def reset_state(self) -> None:
        """Mirrors rl_reset_state() -- call on ball-lost / re-home."""
        self._prev_obs_pos = np.zeros(2, dtype=np.float32)
        self._filt_vel = np.zeros(2, dtype=np.float32)
        self._have_prev_pos = False

    def _forward(self, obs: np.ndarray) -> np.ndarray:
        h1 = np.tanh(obs @ self.w1 + self.b1)
        h2 = np.tanh(h1 @ self.w2 + self.b2)
        out = h2 @ self.w3 + self.b3
        return np.clip(out, -1.0, 1.0)

    def infer(
        self,
        x_mm: float,
        y_mm: float,
        target_x_mm: float,
        target_y_mm: float,
        actual_steps: tuple,
        actual_dt: float,
    ) -> np.ndarray:
        """Mirrors rl_infer(). `actual_steps` must be the motors' REAL current
        step position (e.g. read back from telemetry) -- NOT the last commanded
        target. The network was trained on lagged motor state specifically;
        feeding it the commanded target instead will run it out of distribution.
        Returns int32 step-space targets for motors A/B/C, ready to feed
        directly into a stepper's target_position -- no further conversion."""
        if self._have_prev_pos:
            raw_vx = (x_mm - self._prev_obs_pos[0]) / actual_dt
            raw_vy = (y_mm - self._prev_obs_pos[1]) / actual_dt
        else:
            raw_vx = raw_vy = 0.0

        self._filt_vel[0] = VEL_FILTER_ALPHA * raw_vx + (1 - VEL_FILTER_ALPHA) * self._filt_vel[0]
        self._filt_vel[1] = VEL_FILTER_ALPHA * raw_vy + (1 - VEL_FILTER_ALPHA) * self._filt_vel[1]
        self._prev_obs_pos[:] = (x_mm, y_mm)
        self._have_prev_pos = True

        obs = np.array(
            [
                x_mm,
                y_mm,
                x_mm - target_x_mm,
                y_mm - target_y_mm,
                self._filt_vel[0],
                self._filt_vel[1],
                actual_steps[0],
                actual_steps[1],
                actual_steps[2],
            ],
            dtype=np.float32,
        )

        action = self._forward(obs)
        return np.round(action * MAX_MOTOR_STEP).astype(np.int32)


def self_test() -> None:
    """Structural validation only -- shapes and param count. Does NOT confirm
    numerical agreement with the real STM32 (no hardware available to compare
    against here)."""
    net = ControlNet()
    print(f"Parsed architecture: {net.n_in} -> {net.n_h1} -> {net.n_h2} -> {net.n_out}")
    print(f"Total params: {net.total_params}")
    assert net.total_params == 1475, f"expected 1475 params (~1.5K per AGENTS.md), got {net.total_params}"

    net.reset_state()
    out1 = net.infer(0.0, 0.0, 0.0, 0.0, (0, 0, 0), CONTROL_DT)
    assert out1.shape == (3,)
    assert out1.dtype == np.int32
    out2 = net.infer(5.0, -3.0, 0.0, 0.0, (10, -5, 2), CONTROL_DT)
    assert out2.shape == (3,)
    print(f"Sample outputs (structural check only): {out1}, {out2}")
    print("PASSED -- structural checks only, NOT validated against real STM32 output.")


if __name__ == "__main__":
    self_test()
