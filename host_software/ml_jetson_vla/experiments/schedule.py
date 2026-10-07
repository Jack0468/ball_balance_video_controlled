"""Pure chunk-scheduling and staleness logic. No torch, no lerobot, no pyserial.

A chunk of N steps is consumed at CONTROL_HZ. A replan is triggered when the time left in the
current chunk is below the measured inference estimate plus a margin, so inference overlaps with
playback. A chunk that lands after the previous one has run out is a stall: the arm had no new
command for that gap.
"""

from __future__ import annotations

import collections
import dataclasses
from typing import Any, Optional, Sequence

import numpy as np

from ml_jetson_vla.experiments.serial_protocol import ChunkSafety

CONTROL_HZ: float = 30.0
DEFAULT_MARGIN_MS: float = 40.0  # must exceed one 30 Hz tick (33 ms): the replan check runs once per tick
INFERENCE_WINDOW: int = 8  # the estimate is the max over this many recent measurements (conservative)


@dataclasses.dataclass(frozen=True)
class ChunkPlan:
    option: str
    weights_status: str  # printed in every output: random init vs pretrained vs trained
    actions_deg: np.ndarray  # (N, 3), raw model output, never clamped
    safety: ChunkSafety
    inference_ms: float

    @property
    def accepted(self) -> bool:
        return self.safety.accepted


@dataclasses.dataclass(frozen=True)
class ReplanRecord:
    index: int
    trigger_s: float
    land_s: float
    inference_ms: float
    target_age_ms: Optional[float]
    new_target_arrived: bool
    accepted: bool
    stall: bool
    stall_gap_ms: float
    remaining_ms_at_trigger: float


def target_age_ms(target_done_s: float, used_s: float) -> float:
    age = (used_s - target_done_s) * 1000.0
    if age < 0.0:
        raise ValueError(f"target used at {used_s} before it was done at {target_done_s}")
    return age


class ChunkScheduler:
    def __init__(
        self,
        control_hz: float = CONTROL_HZ,
        margin_ms: float = DEFAULT_MARGIN_MS,
        window: int = INFERENCE_WINDOW,
    ) -> None:
        if control_hz <= 0.0 or margin_ms < 0.0 or window < 1:
            raise ValueError("control_hz must be > 0, margin_ms >= 0, window >= 1")
        self._hz: float = control_hz
        self._margin_ms: float = margin_ms
        self._inference_ms: collections.deque[float] = collections.deque(maxlen=window)
        self._chunk_start_s: Optional[float] = None
        self._chunk_end_s: Optional[float] = None
        self._trigger_s: Optional[float] = None
        self._trigger_remaining_ms: float = 0.0
        self.records: list[ReplanRecord] = []

    @property
    def chunk_start_s(self) -> Optional[float]:
        return self._chunk_start_s

    @property
    def chunk_end_s(self) -> Optional[float]:
        return self._chunk_end_s

    @property
    def inference_pending(self) -> bool:
        return self._trigger_s is not None

    def inference_estimate_ms(self) -> float:
        return max(self._inference_ms) if self._inference_ms else 0.0

    def remaining_s(self, now_s: float) -> float:
        if self._chunk_end_s is None:
            return 0.0
        return max(0.0, self._chunk_end_s - now_s)

    def should_replan(self, now_s: float) -> bool:
        if self._trigger_s is not None:
            return False
        return self.remaining_s(now_s) * 1000.0 < self.inference_estimate_ms() + self._margin_ms

    def begin_inference(self, now_s: float) -> None:
        if self._trigger_s is not None:
            raise RuntimeError("inference already pending")
        self._trigger_s = now_s
        self._trigger_remaining_ms = self.remaining_s(now_s) * 1000.0

    def land(
        self,
        land_s: float,
        plan: ChunkPlan,
        target_age: Optional[float],
        new_target_arrived: bool,
    ) -> ReplanRecord:
        if self._trigger_s is None:
            raise RuntimeError("land() without begin_inference()")
        if land_s < self._trigger_s:
            raise ValueError("chunk landed before its inference was triggered")
        n_send = int(plan.actions_deg.shape[0]) if plan.accepted else 0  # rejected = nothing to play
        had_chunk = self._chunk_end_s is not None
        stall = (not had_chunk) or land_s > self._chunk_end_s
        gap_ms = max(0.0, land_s - self._chunk_end_s) * 1000.0 if had_chunk else 0.0
        start_s = max(land_s, self._chunk_end_s) if had_chunk else land_s
        self._chunk_start_s = start_s
        self._chunk_end_s = start_s + n_send / self._hz
        self._inference_ms.append(plan.inference_ms)
        record = ReplanRecord(
            index=len(self.records),
            trigger_s=self._trigger_s,
            land_s=land_s,
            inference_ms=plan.inference_ms,
            target_age_ms=target_age,
            new_target_arrived=new_target_arrived,
            accepted=plan.accepted,
            stall=stall,
            stall_gap_ms=gap_ms,
            remaining_ms_at_trigger=self._trigger_remaining_ms,
        )
        self.records.append(record)
        self._trigger_s = None
        return record


def summarize_replans(records: Sequence[ReplanRecord]) -> dict[str, Any]:
    if not records:
        raise ValueError("no replan records to summarize")
    n = len(records)
    inf = np.asarray([r.inference_ms for r in records], dtype=np.float64)
    ages = np.asarray([r.target_age_ms for r in records if r.target_age_ms is not None], dtype=np.float64)
    out: dict[str, Any] = {
        "n_replans": n,
        "n_stalls": int(sum(r.stall for r in records)),
        "stall_fraction": float(sum(r.stall for r in records)) / n,
        "first_chunk_was_stall": bool(records[0].stall),  # always true: nothing plays before the first chunk
        "stall_gap_ms_total": float(sum(r.stall_gap_ms for r in records)),
        "n_rejected_chunks": int(sum(not r.accepted for r in records)),
        "n_new_target_arrivals": int(sum(r.new_target_arrived for r in records)),
        "inference_ms_p50": float(np.percentile(inf, 50)),
        "inference_ms_p90": float(np.percentile(inf, 90)),
        "inference_ms_max": float(inf.max()),
        "n_with_target": int(ages.size),
    }
    if ages.size:
        out["target_age_ms_p50"] = float(np.percentile(ages, 50))
        out["target_age_ms_p90"] = float(np.percentile(ages, 90))
        out["target_age_ms_max"] = float(ages.max())
    else:
        out["target_age_ms_p50"] = out["target_age_ms_p90"] = out["target_age_ms_max"] = None
    return out
