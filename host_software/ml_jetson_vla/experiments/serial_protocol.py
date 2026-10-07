"""Pure Python mirror of the angle-control serial protocol in
firmware/stm32_jetson_vla_angle_control/BallBalancingBot/AngleStepControl.cpp.

Downlink (Jetson -> MCU): "A,<theta_a>,<theta_b>,<theta_c>", degrees relative to each motor's
homed zero. The firmware REJECTS (never clamps) any line with a value outside +/-MAX_MOTOR_ANGLE_DEG,
so this module rejects the whole chunk before anything is sent.

Uplink (MCU -> Jetson): "T,<seq>,<mcu_ms>,<touch_x>,<touch_y>,<valid>,<a>,<b>,<c>", touch in
hundredths of a mm (TouchProbe.h), plate-centre origin. Lines starting with '#' are comments.

No pyserial import here: the caller owns the port. The firmware is not flashed or tested by this module.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from typing import Optional

import numpy as np

MAX_MOTOR_ANGLE_DEG: float = 11.025  # AngleStepControl.cpp: 98 steps * 360/3200 deg
ANGLE_DECIMALS: int = 4  # 11.0250 is exact, so the formatted limit equals the firmware limit
N_MOTORS: int = 3
LEVEL_LINE: str = "A,0.0000,0.0000,0.0000"  # firmware levels the plate on stale input
LOG: logging.Logger = logging.getLogger(__name__)


class AngleOutOfRangeError(ValueError):
    """A value is non-finite or outside +/-MAX_MOTOR_ANGLE_DEG."""


@dataclasses.dataclass(frozen=True)
class ChunkSafety:
    accepted: bool
    n_steps: int
    n_violations: int
    max_abs_deg: float  # over finite values; NaN when there are none
    reason: str


@dataclasses.dataclass(frozen=True)
class UplinkSample:
    seq: int
    mcu_ms: int
    touch_x_mm: float
    touch_y_mm: float
    valid: bool  # True = real contact; False = no ball on the plate
    steps: tuple[int, int, int]


def is_angle_in_range(value: float) -> bool:
    return math.isfinite(value) and abs(value) <= MAX_MOTOR_ANGLE_DEG


def format_angle_line(theta_a: float, theta_b: float, theta_c: float) -> str:
    angles = (float(theta_a), float(theta_b), float(theta_c))
    bad = [i for i, v in enumerate(angles) if not is_angle_in_range(v)]
    if bad:
        raise AngleOutOfRangeError(
            f"motor index(es) {bad} outside +/-{MAX_MOTOR_ANGLE_DEG} deg or non-finite: {angles}"
        )
    return "A," + ",".join(f"{v:.{ANGLE_DECIMALS}f}" for v in angles)


def validate_chunk_angles(chunk: np.ndarray) -> ChunkSafety:
    arr = np.asarray(chunk, dtype=np.float64)  # float64 so the boundary is judged at full precision
    if arr.ndim != 2 or arr.shape[1] != N_MOTORS:
        raise ValueError(f"chunk must have shape (N, {N_MOTORS}), got {arr.shape}")
    n_steps = int(arr.shape[0])
    if n_steps == 0:
        return ChunkSafety(False, 0, 0, float("nan"), "empty chunk")
    finite = np.isfinite(arr)
    in_range = finite & (np.abs(arr) <= MAX_MOTOR_ANGLE_DEG)
    n_viol = int(arr.size - int(in_range.sum()))
    max_abs = float(np.max(np.abs(arr[finite]))) if finite.any() else float("nan")
    if n_viol == 0:
        reason = "ok"
    else:
        reason = (f"{n_viol} value(s) outside +/-{MAX_MOTOR_ANGLE_DEG} deg or non-finite; "
                  f"whole chunk rejected, nothing sent")
    return ChunkSafety(n_viol == 0, n_steps, n_viol, max_abs, reason)


def release_chunk_lines(chunk: np.ndarray) -> Optional[list[str]]:
    """Serial lines for an accepted chunk, or None (logged) when the chunk is rejected."""
    safety = validate_chunk_angles(chunk)
    if not safety.accepted:
        LOG.warning("chunk rejected, nothing sent: %s", safety.reason)
        return None
    arr = np.asarray(chunk, dtype=np.float64)
    return [format_angle_line(row[0], row[1], row[2]) for row in arr]


def parse_uplink_line(line: str) -> Optional[UplinkSample]:
    text = line.strip()
    if not text or text.startswith("#") or not text.startswith("T,"):
        return None
    parts = text.split(",")
    if len(parts) != 9 or parts[5] not in ("0", "1"):
        return None
    try:
        seq = int(parts[1])
        mcu_ms = int(parts[2])
        touch_x_mm = int(parts[3]) / 100.0  # firmware sends hundredths of a mm
        touch_y_mm = int(parts[4]) / 100.0
        steps = (int(parts[6]), int(parts[7]), int(parts[8]))
    except ValueError:
        return None
    return UplinkSample(seq, mcu_ms, touch_x_mm, touch_y_mm, parts[5] == "1", steps)
