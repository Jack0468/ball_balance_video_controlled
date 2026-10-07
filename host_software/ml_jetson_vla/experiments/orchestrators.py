"""Target orchestrators: free-text command -> TargetRequest. The fast layer never sees a colour word.

OPTION S (StateMachineOrchestrator): a fixed map from the recognised vocabulary. Colour commands
resolve to a marker label for Qwen grounding on the current frame; directional/hold/stop resolve
to a fixed telemetry-mm target (a nudged hold point, as in src/state_machine.py).

OPTION L (LanguageOrchestrator): interface and wiring only. No language model is selected or loaded.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Optional, Protocol

import numpy as np

from ml_jetson_vla.core.minimal_vlm_policy import (
    COLOR_COMMANDS,
    DIRECTIONAL_COMMANDS,
    PLATFORM_HEIGHT_MM,
    PLATFORM_WIDTH_MM,
)

# Mirrors src/state_machine.py (_NUDGE_X/_NUDGE_Y = 0.15 * platform, clamp = 0.90 * half-range).
# Not imported: that module pulls in the live audio/vision stack.
NUDGE_FRACTION: float = 0.15
CLAMP_FRACTION: float = 0.90
# Camera is rotated 180 deg relative to the platform, so camera-left is physical +X (state_machine.py).
NUDGE_SIGNS: dict[str, tuple[float, float]] = {
    "forward": (0.0, 1.0),
    "backward": (0.0, -1.0),
    "left": (1.0, 0.0),
    "right": (-1.0, 0.0),
}
HOLD_COMMANDS: frozenset[str] = frozenset({"hold", "stop"})  # state_machine.py treats stop as hold
LLM_NOT_IMPLEMENTED_MESSAGE: str = (
    "OPTION L (act_fast_llm, language orchestrator) is not implemented: no language model is "
    "selected or loaded in this codebase. Supply a parse_fn that maps free text to one of "
    f"{sorted(COLOR_COMMANDS.values())} to wire it in; nothing is loaded by default."
)


@dataclasses.dataclass(frozen=True)
class TargetRequest:
    instruction: str
    marker_label: Optional[str]  # grounding prompt label, for colour commands
    fixed_target_mm: Optional[tuple[float, float]]  # telemetry mm, for directional/hold/stop
    fixed_action: Optional[str]

    def __post_init__(self) -> None:
        if (self.marker_label is None) == (self.fixed_target_mm is None):
            raise ValueError("exactly one of marker_label / fixed_target_mm must be set")


class TargetOrchestrator(Protocol):
    def resolve(self, instruction: str, frame: np.ndarray, state: np.ndarray) -> TargetRequest:
        """`state`: observation.state, (touch_x_mm, touch_y_mm), telemetry frame. `frame` is BGR."""
        ...


def _clamp_mm(x: float, y: float) -> tuple[float, float]:
    cx = CLAMP_FRACTION * (PLATFORM_WIDTH_MM / 2.0)
    cy = CLAMP_FRACTION * (PLATFORM_HEIGHT_MM / 2.0)
    return max(-cx, min(cx, x)), max(-cy, min(cy, y))


class StateMachineOrchestrator:
    def __init__(self) -> None:
        self._hold_mm: Optional[tuple[float, float]] = None

    def reset(self) -> None:
        self._hold_mm = None

    def resolve(self, instruction: str, frame: np.ndarray, state: np.ndarray) -> TargetRequest:
        if instruction in COLOR_COMMANDS:
            self._hold_mm = None  # leaving hold mode; the next hold re-seeds from the ball
            return TargetRequest(instruction, COLOR_COMMANDS[instruction], None, None)
        if instruction not in DIRECTIONAL_COMMANDS:
            raise ValueError(f"unrecognised instruction {instruction!r}")
        if self._hold_mm is None:
            # Seed from the current ball position, as state_machine.py does on entering hold.
            self._hold_mm = (float(state[0]), float(state[1]))
        if instruction in NUDGE_SIGNS:
            sx, sy = NUDGE_SIGNS[instruction]
            hx, hy = self._hold_mm
            hx += sx * NUDGE_FRACTION * PLATFORM_WIDTH_MM
            hy += sy * NUDGE_FRACTION * PLATFORM_HEIGHT_MM
            self._hold_mm = _clamp_mm(hx, hy)
            action = f"nudge_{instruction}"
        else:
            action = "hold"
        assert self._hold_mm is not None
        return TargetRequest(instruction, None, self._hold_mm, action)


class LanguageOrchestrator:
    def __init__(self, parse_fn: Optional[Callable[[str], str]] = None) -> None:
        self._parse_fn = parse_fn

    def ensure_available(self) -> None:
        if self._parse_fn is None:
            raise NotImplementedError(LLM_NOT_IMPLEMENTED_MESSAGE)

    def resolve(self, instruction: str, frame: np.ndarray, state: np.ndarray) -> TargetRequest:
        self.ensure_available()
        assert self._parse_fn is not None
        label = self._parse_fn(instruction)
        if label not in COLOR_COMMANDS.values():
            raise ValueError(f"parse_fn returned {label!r}; expected one of {sorted(COLOR_COMMANDS.values())}")
        return TargetRequest(instruction, label, None, None)
