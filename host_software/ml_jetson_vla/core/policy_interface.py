"""Model-agnostic policy wrapper, per docs/ARCHITECTURE.md's documented module layout:
`act(image, instruction, state) -> command`. Exists so `runtime/run_jetson_standalone.py`
(and, later, arm 2/3's own entry points) can be driven by the same eval-harness shape
regardless of which model class sits underneath -- swapping Track 1 (small-class expert
pipeline) for Track 3 (medium-class) or a future large-VLA arm should only mean swapping
which JetsonExpertPolicy subclass/config gets constructed, not touching the runtime loop.

NOTE: `host_software/ml_multimodal/` does not currently have an equivalent wrapper --
`run_eval_baseline_vla.py`/`run_eval_our_vla.py` call `RT1LiteVLA.forward(img, cmd_idx,
state)` directly. ARCHITECTURE.md's "shared shape with ml_multimodal's equivalent" is
aspirational, not a currently-existing thing to conform to; this module defines the shape
going forward per that doc, it does not mirror pre-existing code.
"""

from __future__ import annotations

import dataclasses
from typing import Optional, Protocol

import numpy as np


@dataclasses.dataclass
class PolicyCommand:
    """Everything the runtime loop needs to act on one policy step. `target_x_mm` /
    `target_y_mm` are always populated (what the state machine wants the ball to reach);
    `step_targets` is populated only once a policy computes raw step-space targets itself
    (Phase B / control-net-on-Jetson) -- None here means "let the STM32's own on-device
    control net decide," which is Track 1's Phase-A behavior."""

    target_x_mm: float
    target_y_mm: float
    step_targets: Optional[tuple] = None  # (stepA, stepB, stepC), Phase B only
    debug: Optional[dict] = None


class Policy(Protocol):
    """Structural interface every arm's policy implementation satisfies. Not an ABC on
    purpose -- Track 1's implementation composes several existing ONNX sessions rather
    than being one model, so there's no single natural base class to inherit from."""

    def act(
        self,
        image: np.ndarray,
        instruction: Optional[str],
        state: dict,
    ) -> PolicyCommand:
        """`image`: BGR frame straight from the camera receiver (pre-warp) -- policies own
        their own preprocessing, since Track 1's ArUco-homography warp and a future VLA's
        raw-frame tokenizer have nothing in common upstream of this call.
        `instruction`: latest recognized command string (e.g. "go_blue"), or None if
        nothing new since the last call -- same shape as `AudioCommandReceiverONNX`'s
        `get_latest_command()` already returns today.
        `state`: free-form dict for whatever closed-loop state a policy needs across calls
        (e.g. Track 1 passes through `TargetStateMachine`/`PredictionGate` state; a future
        control-net-on-Jetson policy would carry `actual_steps` here). Policies that don't
        need cross-call state may ignore it."""
        ...

    def reset(self) -> None:
        """Called on ball-lost / re-home, mirrors ControlNet.reset_state() and
        PredictionGate's AWAITING_BALL re-entry."""
        ...
