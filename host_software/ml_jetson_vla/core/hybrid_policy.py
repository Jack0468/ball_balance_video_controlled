"""Arm 2's two experimental control options, wrapped behind the real `core.policy_interface.Policy`
contract -- not the benchmark/replay-only shape `experiments/run_experiment.py` drives directly.
See `docs/EXPERIMENT_OPTIONS_PLAN_2026_10_07.md`'s status table for what's wired here vs. still
only exercised as a replay/benchmark script.

`HybridQwenActPolicy`: `StateMachineOrchestrator` (or a `LanguageOrchestrator`, once a real
`parse_fn` exists -- both satisfy `experiments.orchestrators.TargetOrchestrator`) resolves the
latest instruction into a `TargetRequest`. Colour commands are handed to a background grounding
worker (`_Grounder`, `QwenVLBackend` via `core.minimal_vlm_policy.MinimalVLMPolicy`, reusing its
prompt/parse/pixel-to-mm pipeline rather than re-deriving it) running on its own slow cadence --
same `TargetStore` split as `deployment/bench_hybrid_qwen_act.py`'s slow loop, imported from there,
not reimplemented. **Periodic re-grounding (2026-10-13 fix, see `DEFAULT_REGROUND_INTERVAL_S`)**:
a colour command is re-submitted to the grounder on that cadence for as long as it stays the held
instruction, not just once when it first arrives -- closes the staleness gap flagged 2026-10-06
(target-age p90 measured 4.2-8.9s with no re-grounding) and
`EXPERIMENT_OPTIONS_PLAN_2026_10_07.md` section 4 "Known limit #2". Directional/hold/stop commands
resolve to a fixed mm target with no grounding call (and clear any held colour command's periodic
re-grounding, since it is no longer the active target). The fast layer (`ActFastLayer`, or
`StubFastLayer` for CPU/offline use) runs in its own
background worker (`_FastWorker`), chunk-scheduled via `experiments.schedule.ChunkScheduler` --
`act()` never blocks on either worker: it serves whatever chunk element is already scheduled and
submits a new replan if one is due, exactly like `bench_hybrid_qwen_act.py`'s two-rate model and
`run_experiment.py`'s (unexercised) `run_live()`, now behind `Policy`.

`SmolVLADirectPolicy`: no grounding call -- `SmolVLAFastLayer` takes the instruction text itself
and does its own grounding internally. Same `_FastWorker`/scheduler/safety-gate machinery.

Safety gate (serial_protocol's reject-whole-chunk-never-clamp rule): every `FastLayer.plan()` call
already runs `serial_protocol.validate_chunk_angles()` internally (`fast_layers._make_plan`) and a
rejected `ChunkPlan.accepted is False` chunk is never adopted as `_FastWorker.active_actions` --
`PolicyCommand.angle_targets_deg` is `None` for every frame until an accepted chunk is actually
playing. `_FastWorker.current_angles()` additionally re-validates the one row it is about to
return through `serial_protocol.format_angle_line()` (which raises on an out-of-range value) as a
last gate right before a value would leave this module, belt-and-suspenders on top of the
whole-chunk check already done upstream.

Honesty contract ("no model loaded"/weights-not-trained, must not silently pretend a trained
result): both policies expose `weights_status` (verbatim from the wrapped `FastLayer`, e.g.
`StubFastLayer`'s "STUB (no model), constant ... deg" or `ActFastLayer`'s
"RANDOM_INIT: no trained ACT checkpoint; actions are not meaningful") and `trained_for_task: bool`
(`True` only when `weights_status` contains `ActFastLayer`'s own "TRAINED checkpoint" substring,
i.e. a real `--checkpoint` was supplied -- `StubFastLayer` and a never-fine-tuned `SmolVLAFastLayer`
are always `False`). Both are also echoed into every `PolicyCommand.debug` dict, since
`policy_interface.py`'s `Policy` protocol has no dedicated metadata field for this and `debug` is
the one it does allow.

Touch-state contract (not covered by `policy_interface.py`'s own docstring, decided here):
`state["touch_mm"]` is `Optional[tuple[float, float]]`, the latest `TouchProbe` reading
(`experiments.serial_protocol.UplinkSample`), mm, plate-centre origin -- NOT the vision-CNN ball
position `JetsonExpertPolicy` computes from the camera frame. The caller owns polling the serial
uplink (see `run_experiment.py`'s `run_live()` `uplink()` thread) and feeding the latest sample in
through `state` each call, matching `policy_interface.py`'s "free-form dict ... policies that don't
need cross-call state may ignore it" allowance -- these two policies DO need a fresh reading every
call, so the key they read is documented here rather than inventing a new `Policy` method.
`act()` without a finite `touch_mm` never submits a replan (no state to condition the fast layer
on) and returns whatever chunk is already playing, if any.

SMOLVLA_DIRECT's target input: `fast_layers.SmolVLAFastLayer.plan()` takes a `target_mm` parameter
structurally (shared `FastLayer` interface) but never reads it in its body -- SmolVLA grounds
itself on the instruction text. `SmolVLADirectPolicy` passes the fixed placeholder `(0.0, 0.0)`,
the same convention `run_experiment.py`'s `run_live()` uses for this option ("smolvla_direct: no
target input in the wiring"), and reports `target_x_mm`/`target_y_mm` as NaN at the `Policy`
boundary rather than exposing that internal placeholder as if it were a real resolved target --
same NaN-on-unknown convention `core/minimal_vlm_policy.py` already uses.

Instruction text form: `Policy.act()`'s `instruction` is documented (`policy_interface.py`) as the
raw recognised vocabulary string (e.g. `"go_blue"`, underscores, `AudioCommandReceiverONNX`'s own
form) -- that is the form `TargetOrchestrator.resolve()` and `COLOR_COMMANDS`/`DIRECTIONAL_COMMANDS`
(`minimal_vlm_policy.py`, `orchestrators.py`) key on, so it is passed through to `resolve()` and to
the grounding request unchanged. The fast layer and the grounding prompt both want natural-language
text instead (`"go blue"`) -- `run_experiment.py`'s own `command_to_text()` (`str.replace("_", " ")`)
is reused (imported, not re-derived) to convert right before each of those two calls, matching
`run_experiment.py`'s `run_replay()`/`run_live()` convention exactly (`orch.resolve(cmd, ...)` on the
raw form, `layer.plan(..., instruction=command_to_text(cmd))` on the converted one).

Found while building this (not fixed here, out of this module's scope): `run_experiment.py`'s
`run_live()` queues `req.marker_label` (e.g. "green marker") as the grounding worker's
`instruction` argument, but `MinimalVLMPolicy.build_prompt()` matches instructions against
`COLOR_COMMANDS`' KEYS (e.g. "go_green") -- as written, every colour command there falls through
to the directional "ball" fallback. `_Grounder` below queues `req.instruction` (the original
command) instead, which is what `build_prompt()` expects.

`HybridSmallCnnActPolicy` (2026-10-13): a third `Policy`, sibling to the two above, built as a
**debugging/staging configuration -- NOT a new official Arm 1 vs Arm 2 comparison arm** (see
`docs/ARCHITECTURE.md`'s 3-arm framing; this combination is not one of those three). ACT
(`ActFastLayer`) isn't trained yet, and testing/training its own control behaviour against Qwen's
comparatively noisy (~60-93% hit@20mm, measured this session) and slow (~1.7-2s/call, GPU-shared
with ACT) grounding confounds two unvalidated things at once. This policy instead grounds colour
commands with `core.small_cnn_grounder.SmallCnnGroundingPolicy` -- Track 1's already-validated,
sub-10ms small-class CNN + ArUco + `MarkerClassifier` pipeline (`JetsonExpertPolicy`'s own
production pipeline, reused) -- giving ACT a fast, reliable grounding signal to be tested/trained
against in isolation. Made possible by generalising `_Grounder` below to wrap ANY `Policy`-shaped
grounding implementation (duck-typed: `.act()` returning a `PolicyCommand` plus a `.last_debug`
dict with a `parse_ok` key), not just `MinimalVLMPolicy` specifically -- `SmallCnnGroundingPolicy`
satisfies that same shape (see its own module docstring), so `_Grounder`'s existing
threading/queue/`TargetStore` machinery, and `_FastWorker`/`ChunkScheduler` on the fast-layer side,
are reused completely unchanged. `HybridSmallCnnActPolicy` itself subclasses `HybridQwenActPolicy`
and overrides only `__init__` (via the shared `_init_common()` helper both now use) to swap which
`Policy` backs the `_Grounder` -- `act()`/`reset()`/`close()`/`replan_records` never reference
anything Qwen/`VLMBackend`-specific, only `self._grounder` (a plain `_Grounder`), so no other code
needed to change. This module itself still does NOT import `small_cnn_grounder.py` (which pulls in
`onnxruntime`/`cv2`) -- callers construct the `SmallCnnGroundingPolicy` and pass it in, keeping
this module's own import surface unchanged (see `ImportIsolationTests`, which only ever checked
torch/lerobot and still only needs to).
"""

from __future__ import annotations

import dataclasses
import queue
import threading
import time
from typing import Any, Callable, Optional, Union

import numpy as np

from ml_jetson_vla.core.minimal_vlm_policy import MinimalVLMPolicy
from ml_jetson_vla.core.policy_interface import Policy, PolicyCommand
from ml_jetson_vla.core.vlm_backends import VLMBackend
from ml_jetson_vla.deployment.bench_hybrid_qwen_act import TargetStore
from ml_jetson_vla.experiments.fast_layers import FastLayer
from ml_jetson_vla.experiments.orchestrators import (
    LanguageOrchestrator,
    StateMachineOrchestrator,
    TargetOrchestrator,
    TargetRequest,
)
from ml_jetson_vla.experiments.schedule import (
    CONTROL_HZ,
    DEFAULT_MARGIN_MS,
    ChunkScheduler,
    ReplanRecord,
    target_age_ms,
)
from ml_jetson_vla.experiments.run_experiment import command_to_text
from ml_jetson_vla.experiments.serial_protocol import AngleOutOfRangeError, format_angle_line

_QUEUE_POLL_S: float = 0.05
_THREAD_JOIN_TIMEOUT_S: float = 2.0
TRAINED_MARKER: str = "TRAINED checkpoint"  # exact substring ActFastLayer sets for a real checkpoint

# Periodic re-grounding cadence (2026-10-13 fix for the staleness gap flagged 2026-10-06 and
# EXPERIMENT_OPTIONS_PLAN_2026_10_07.md section 4 "Known limit #2"): a held colour command used to
# ground exactly once, then never again, letting target age grow unbounded for as long as the
# command stayed held (measured p90 4.2-8.9s across real sessions). Re-requesting on this cadence
# instead bounds steady-state target age by roughly max(interval, grounding backend latency):
# Qwen's own measured ~1.7-2s/call (`docs/PROJECT_LOGBOOK.md` 09/10/2026) dominates when it is the
# backend, landing a fresh result roughly every ~1.7-2s and keeping age under this project's
# proposed <=2000ms criterion (`EXPERIMENT_OPTIONS_PLAN_2026_10_07.md` section 3); the 2026-10-13
# small-CNN grounder (<10ms/call) is instead bounded by this interval itself. `_Grounder.request()`
# is already non-blocking and drops a request if the previous one is still in flight (unchanged),
# so firing this on every tick once the interval has elapsed is safe and never stalls `act()`.
DEFAULT_REGROUND_INTERVAL_S: float = 1.0


def _is_trained_for_task(weights_status: str) -> bool:
    return TRAINED_MARKER in weights_status


class _FastWorker:
    """Background thread wrapping one `FastLayer.plan()` call per request: the fast loop's own
    half of the slow/fast split (`bench_hybrid_qwen_act.py`'s two threads, `run_experiment.py`'s
    `run_live()` `fast_worker()`/`fast_q`/`fast_out` pairing) -- reused as a pattern, not
    reimplemented inline in each `Policy`, so both `HybridQwenActPolicy` and
    `SmolVLADirectPolicy` share one scheduler/safety-gate implementation. `submit_if_due()` and
    `poll_landed()` are both non-blocking; the caller's `act()` never waits on this thread."""

    def __init__(
        self, fast_layer: FastLayer, control_hz: float = CONTROL_HZ, margin_ms: float = DEFAULT_MARGIN_MS
    ) -> None:
        self.fast_layer: FastLayer = fast_layer
        self.scheduler: ChunkScheduler = ChunkScheduler(control_hz, margin_ms)
        self._control_hz: float = control_hz
        self._in_q: "queue.Queue[tuple[Any, ...]]" = queue.Queue(maxsize=1)
        self._out_q: "queue.Queue[tuple[Any, Optional[float], bool, Optional[float]]]" = queue.Queue()
        self._stop = threading.Event()
        self._errors: list[str] = []
        self.active_actions: Optional[np.ndarray] = None
        self.replan_pending: bool = False
        self.n_rejected: int = 0
        self.n_consecutive_rejects: int = 0
        self.last_rejection_reason: Optional[str] = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    @property
    def errors(self) -> list[str]:
        return list(self._errors)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                frame, target_mm, state_mm, instruction, age, new_arrived, target_done_s = self._in_q.get(
                    timeout=_QUEUE_POLL_S
                )
            except queue.Empty:
                continue
            try:
                plan = self.fast_layer.plan(target_mm, state_mm, frame, instruction=instruction)
            except Exception as exc:  # a dead fast worker must be visible, never silently stall forever
                self._errors.append(f"{type(exc).__name__}: {exc}")
                return
            self._out_q.put((plan, age, new_arrived, target_done_s))

    def submit_if_due(
        self,
        now: float,
        frame: np.ndarray,
        target_mm: tuple,
        state_mm: tuple,
        instruction: Optional[str],
        age: Optional[float],
        new_arrived: bool,
        target_done_s: Optional[float] = None,
    ) -> bool:
        """`target_done_s` is the target timestamp actually being fed to `fast_layer.plan()` for
        THIS submission -- captured here, at submission time, and carried through the queues so
        that `poll_landed()` can report back exactly what was used for a given chunk rather than
        whatever the store's `latest()` happens to be when that chunk lands later (finding 2:
        using the landing-time value let a brand-new target be mistaken for already-used)."""
        if self.replan_pending or not self.scheduler.should_replan(now):
            return False
        self.scheduler.begin_inference(now)
        self.replan_pending = True
        try:
            # Copy the frame before handing it to the background thread (same reason
            # `_Grounder.request()` and `run_experiment.py`'s fast/ground queues already do this):
            # the caller may reuse a preallocated capture buffer on the very next tick.
            self._in_q.put_nowait((frame.copy(), target_mm, state_mm, instruction, age, new_arrived, target_done_s))
            return True
        except queue.Full:
            # should not happen (replan_pending gates this), but never block act() on it
            return True

    def poll_landed(self, now: float) -> list[tuple[ReplanRecord, Optional[float]]]:
        """Returns `(record, target_done_s)` pairs, where `target_done_s` is the submission-time
        value passed through `submit_if_due()` for that specific chunk (see its docstring) --
        not a re-read of the store's current state at landing time."""
        landed: list[tuple[ReplanRecord, Optional[float]]] = []
        while True:
            try:
                plan, age, new_arrived, target_done_s = self._out_q.get_nowait()
            except queue.Empty:
                break
            self.replan_pending = False
            record = self.scheduler.land(now, plan, age, new_arrived)
            if plan.accepted:
                self.active_actions = plan.actions_deg
                self.n_consecutive_rejects = 0
            else:
                self.active_actions = None
                self.n_rejected += 1
                self.n_consecutive_rejects += 1
                self.last_rejection_reason = plan.safety.reason
            landed.append((record, target_done_s))
        return landed

    def current_angles(self, now: float) -> Optional[tuple]:
        if self.active_actions is None or self.scheduler.chunk_start_s is None:
            return None
        step = int(np.floor((now - self.scheduler.chunk_start_s) * self._control_hz))
        if not (0 <= step < self.active_actions.shape[0]):
            return None
        row = self.active_actions[step]
        try:
            format_angle_line(*row)  # last gate: raises on an out-of-range value before it leaves this module
        except AngleOutOfRangeError:
            return None
        return (float(row[0]), float(row[1]), float(row[2]))

    def close(self, timeout: float = _THREAD_JOIN_TIMEOUT_S) -> None:
        self._stop.set()
        self._thread.join(timeout=timeout)


class _Grounder:
    """Background grounding worker: one `Policy`-shaped grounding implementation (originally
    always a `MinimalVLMPolicy` wrapping a `VLMBackend`, e.g. `QwenVLBackend`; generalised
    2026-10-13 to accept any such implementation -- see `core.small_cnn_grounder
    .SmallCnnGroundingPolicy` for the second one, built so ACT can be tested against fast, reliable
    grounding instead of Qwen's -- module docstring has the full rationale) on its own slow
    cadence, writing results into a `TargetStore` -- the exact `TargetStore`/slow-loop split
    `deployment/bench_hybrid_qwen_act.py` already defines, imported here rather than re-derived.
    `request()` is non-blocking and drops a request rather than blocking `act()` if the previous
    one is still in flight (matches `run_experiment.py`'s `run_live()` `ground_q.put_nowait()`
    behaviour). The only contract required of `grounding_policy`: `.act(frame, instruction, state)
    -> PolicyCommand` (the `Policy` protocol), plus a `.last_debug` dict carrying a `parse_ok` key
    afterwards (both `MinimalVLMPolicy` and `SmallCnnGroundingPolicy` already follow this
    convention, duck-typed, same as `JetsonExpertPolicy`'s own `last_debug` attribute)."""

    def __init__(self, grounding_policy: Policy) -> None:
        self.policy: Policy = grounding_policy
        self.store: TargetStore = TargetStore()
        self._q: "queue.Queue[tuple[np.ndarray, str]]" = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._errors: list[str] = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    @property
    def errors(self) -> list[str]:
        return list(self._errors)

    def request(self, frame: np.ndarray, instruction: str) -> bool:
        try:
            self._q.put_nowait((frame.copy(), instruction))
            return True
        except queue.Full:
            return False

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                frame, instruction = self._q.get(timeout=_QUEUE_POLL_S)
            except queue.Empty:
                continue
            t0 = time.monotonic()
            try:
                command = self.policy.act(frame, instruction, state={})
            except Exception as exc:
                self._errors.append(f"{type(exc).__name__}: {exc}")
                return
            latency = time.monotonic() - t0
            debug = self.policy.last_debug
            ok = bool(debug.get("parse_ok")) and np.isfinite(command.target_x_mm) and np.isfinite(command.target_y_mm)
            if not ok:
                self.store.record_parse_failure(latency)
                continue
            self.store.put(command.target_x_mm, command.target_y_mm, instruction, latency)

    def close(self, timeout: float = _THREAD_JOIN_TIMEOUT_S) -> None:
        self._stop.set()
        self._thread.join(timeout=timeout)


def _resolve_target_request(
    orchestrator: TargetOrchestrator, instruction: str, image: np.ndarray, touch_mm: tuple
) -> tuple[Optional[TargetRequest], Optional[str]]:
    """Returns `(request_or_None, not_implemented_message_or_None)` -- a `LanguageOrchestrator`
    with no `parse_fn` wired in raises `NotImplementedError` on every call; callers must not let
    that crash `act()`, since the whole point of accepting a `LanguageOrchestrator`-compatible
    slot is that it can sit there unconfigured until a real one exists."""
    try:
        return orchestrator.resolve(instruction, image, np.asarray(touch_mm, dtype=np.float64)), None
    except NotImplementedError as exc:
        return None, str(exc)


class HybridQwenActPolicy(Policy):
    """ACT_FAST (fast layer) + option-S/L orchestrator + Qwen grounding, behind `Policy`. See
    module docstring for the full contract. `__init__` only builds the `_Grounder` (Qwen-specific,
    via `MinimalVLMPolicy`); everything else is shared with `HybridSmallCnnActPolicy` through
    `_init_common()` below, since `act()`/`reset()`/`close()` never reference anything
    Qwen/`VLMBackend`-specific -- only `self._grounder` (a plain, grounder-policy-agnostic
    `_Grounder`)."""

    def __init__(
        self,
        fast_layer: FastLayer,
        orchestrator: Optional[Union[StateMachineOrchestrator, LanguageOrchestrator]] = None,
        grounding_backend: Optional[VLMBackend] = None,
        pixel_to_mm: Optional[Callable[[float, float], tuple]] = None,
        grounding_prompt_variant: str = "baseline",
        control_hz: float = CONTROL_HZ,
        margin_ms: float = DEFAULT_MARGIN_MS,
        reground_interval_s: float = DEFAULT_REGROUND_INTERVAL_S,
    ) -> None:
        grounder: Optional[_Grounder] = None
        if grounding_backend is not None:
            grounding_policy = MinimalVLMPolicy(
                grounding_backend, pixel_to_mm=pixel_to_mm, prompt_variant=grounding_prompt_variant
            )
            grounder = _Grounder(grounding_policy)
        self._init_common(fast_layer, orchestrator, grounder, control_hz, margin_ms, reground_interval_s)

    def _init_common(
        self,
        fast_layer: FastLayer,
        orchestrator: Optional[Union[StateMachineOrchestrator, LanguageOrchestrator]],
        grounder: Optional[_Grounder],
        control_hz: float,
        margin_ms: float,
        reground_interval_s: float = DEFAULT_REGROUND_INTERVAL_S,
    ) -> None:
        """Shared construction tail for `HybridQwenActPolicy` and `HybridSmallCnnActPolicy` --
        everything downstream of "what `Policy` backs the `_Grounder`, if any". Not part of either
        class's public surface."""
        self._orchestrator: Union[StateMachineOrchestrator, LanguageOrchestrator] = (
            orchestrator if orchestrator is not None else StateMachineOrchestrator()
        )
        self._fast: _FastWorker = _FastWorker(fast_layer, control_hz, margin_ms)
        self._grounder: Optional[_Grounder] = grounder
        self.weights_status: str = fast_layer.weights_status
        self.trained_for_task: bool = _is_trained_for_task(fast_layer.weights_status)
        self._fixed_target_mm: Optional[tuple] = None
        self._fixed_target_done_s: Optional[float] = None
        self._used_done_s: float = -float("inf")
        self._last_instruction: Optional[str] = None
        self._instruction_text: Optional[str] = None  # command_to_text() form, for the fast layer
        self._reground_interval_s: float = reground_interval_s
        # The instruction currently being grounded on a periodic cadence (the raw vocabulary form,
        # e.g. "go_red" -- `_Grounder.request()`'s expected form), or None while no colour command
        # is held (directional/hold/stop, or nothing yet). Set on every colour-command resolve,
        # cleared on a directional/hold/stop resolve or `reset()`. See `DEFAULT_REGROUND_INTERVAL_S`.
        self._active_grounding_instruction: Optional[str] = None
        self._last_ground_request_s: float = -float("inf")
        self.last_debug: dict = {}

    @property
    def replan_records(self) -> list[ReplanRecord]:
        """Read-only access to the fast worker's `ChunkScheduler.records`, for a replay/eval
        driver to feed into `experiments.schedule.summarize_replans()` -- the same stats
        `run_experiment.py`'s replay already reports, no new metric invented."""
        return self._fast.scheduler.records

    @property
    def worker_errors(self) -> list[str]:
        """Combined errors from the background `_FastWorker` and `_Grounder` threads (either
        catches an exception by appending here and then exiting -- see their `_run()` docstrings/
        comments -- "a dead fast worker must be visible, never silently stall forever"). That
        promise was not actually kept by any caller until this property existed: a real incident
        (2026-10-13) had `_FastWorker` die on its first real chunk (a chunk-size mismatch between
        a trained checkpoint and the caller's `--chunk-len`) and no driver noticed -- three live-
        hardware runs (60s/30s/30s) looked like an indefinite hang with zero chunks ever landing,
        when the worker had actually crashed and exited on tick 1. Callers (`run_policy_live.py`,
        `run_policy_replay.py`) should check this every tick/at the end and treat non-empty as a
        real failure, matching the existing `bench_hybrid_qwen_act.py` convention
        (`if store.errors: raise RuntimeError(...)`)."""
        errors = list(self._fast.errors)
        if self._grounder is not None:
            errors.extend(self._grounder.errors)
        return errors

    def reset(self) -> None:
        if hasattr(self._orchestrator, "reset"):
            self._orchestrator.reset()  # type: ignore[union-attr]
        self._fixed_target_mm = None
        self._fixed_target_done_s = None
        self._used_done_s = -float("inf")
        self._last_instruction = None
        self._instruction_text = None
        self._active_grounding_instruction = None
        self._last_ground_request_s = -float("inf")
        self.last_debug = {}

    def close(self) -> None:
        """Stops the background workers. Not part of the `Policy` protocol -- callers that
        construct one of these should call it on shutdown to avoid leaking threads."""
        self._fast.close()
        if self._grounder is not None:
            self._grounder.close()

    def act(self, image: np.ndarray, instruction: Optional[str], state: dict) -> PolicyCommand:
        now = time.monotonic()
        touch_mm = state.get("touch_mm")
        debug: dict[str, Any] = {
            "fast_layer": self._fast.fast_layer.name,
            "weights_status": self.weights_status,
            "trained_for_task": self.trained_for_task,
        }

        just_resolved_new_instruction = False
        if instruction is not None and instruction != self._last_instruction and touch_mm is not None:
            req, not_impl = _resolve_target_request(self._orchestrator, instruction, image, touch_mm)
            if not_impl is not None:
                debug["orchestrator_not_implemented"] = not_impl
            elif req is not None:
                self._last_instruction = instruction
                self._instruction_text = command_to_text(instruction)
                if req.marker_label is not None:
                    just_resolved_new_instruction = True
                    self._active_grounding_instruction = req.instruction
                    debug["grounding_requested"] = True
                    if self._grounder is None:
                        debug["grounding_unavailable"] = "no grounding_policy configured"
                    elif self._grounder.request(image, req.instruction):
                        self._last_ground_request_s = now
                    else:
                        debug["grounding_queue_full"] = True
                else:
                    assert req.fixed_target_mm is not None
                    # Directional/hold/stop supersedes any colour command currently being held --
                    # stop periodically re-grounding a target that is no longer the active one.
                    self._active_grounding_instruction = None
                    self._fixed_target_mm = req.fixed_target_mm
                    self._fixed_target_done_s = now

        # Periodic re-grounding (see `DEFAULT_REGROUND_INTERVAL_S`): while the same colour command
        # stays held (no new instruction this tick, so the block above did not already just issue
        # a request for it), keep re-requesting on a cadence instead of grounding once and letting
        # the target go stale for as long as the command is held. `_Grounder.request()` is
        # non-blocking and already drops a request if the previous one is still in flight.
        if (
            not just_resolved_new_instruction
            and self._active_grounding_instruction is not None
            and self._grounder is not None
            and touch_mm is not None
            and (now - self._last_ground_request_s) >= self._reground_interval_s
        ):
            if self._grounder.request(image, self._active_grounding_instruction):
                self._last_ground_request_s = now
                debug["regrounding_requested"] = True

        latest = self._grounder.store.latest() if self._grounder is not None else None
        target_xy: Optional[tuple] = None
        target_done_s: Optional[float] = None
        if self._fixed_target_done_s is not None and (latest is None or self._fixed_target_done_s >= latest.done_monotonic_s):
            target_xy, target_done_s = self._fixed_target_mm, self._fixed_target_done_s
        elif latest is not None:
            target_xy, target_done_s = (latest.x_mm, latest.y_mm), latest.done_monotonic_s
        has_target = target_xy is not None

        for record, submitted_target_done_s in self._fast.poll_landed(now):
            # Use the target timestamp actually fed to THIS chunk's plan() call at submission
            # time (finding 2), not `target_done_s` re-read above -- a newer target can have
            # arrived in the store between this chunk's submission and its landing, and that
            # newer target must still be free to trigger its own replan later.
            if record.accepted and record.new_target_arrived and submitted_target_done_s is not None:
                self._used_done_s = max(self._used_done_s, submitted_target_done_s)

        touch_finite = touch_mm is not None and bool(np.all(np.isfinite(touch_mm)))
        if has_target and touch_finite:
            assert target_done_s is not None
            # Re-sample monotonic time right at the point of use and clamp `used_s` to be no
            # earlier than `target_done_s` (finding 1): the grounder thread stamps
            # `done_monotonic_s` with its OWN later `time.monotonic()` call, independently of the
            # `now` captured at the top of this method, so `target_done_s` can legitimately be
            # later than a stale `now` even though no real logic error occurred -- that's a small,
            # harmless positive skew from the cross-thread race, not the same-thread misuse
            # `target_age_ms()`'s own raise exists to catch.
            age = target_age_ms(target_done_s, max(time.monotonic(), target_done_s))
            new_arrived = target_done_s > self._used_done_s
            submitted = self._fast.submit_if_due(
                now, image, target_xy, (float(touch_mm[0]), float(touch_mm[1])),
                self._instruction_text, age, new_arrived, target_done_s,
            )
            debug["replan_submitted"] = submitted

        angles = self._fast.current_angles(now)
        debug["has_target"] = has_target
        debug["n_rejected_chunks"] = self._fast.n_rejected
        debug["last_rejection_reason"] = self._fast.last_rejection_reason
        debug["grounding_slow_calls"] = self._grounder.store.slow_calls if self._grounder is not None else None
        debug["grounding_parse_failures"] = self._grounder.store.parse_failures if self._grounder is not None else None
        self.last_debug = debug

        tx, ty = (float(target_xy[0]), float(target_xy[1])) if has_target else (float("nan"), float("nan"))
        return PolicyCommand(target_x_mm=tx, target_y_mm=ty, angle_targets_deg=angles, debug=debug)


class HybridSmallCnnActPolicy(HybridQwenActPolicy):
    """ACT_FAST (fast layer) + option-S/L orchestrator + the small-class Shared Backbone CNN +
    `MarkerClassifier` pipeline for grounding (`core.small_cnn_grounder.SmallCnnGroundingPolicy`),
    instead of Qwen. See the module docstring's "HybridSmallCnnActPolicy (2026-10-13)" section for
    the full rationale.

    **Debugging/staging configuration for validating ACT in isolation -- NOT a new official Arm 1
    vs Arm 2 comparison arm.** Arm 1 is the small-class expert pipeline running standalone (its own
    audio/state-machine/control-net chain, no ACT involved); Arm 2 is the Qwen-driven policy
    (`HybridQwenActPolicy`/`SmolVLADirectPolicy`). This class is a third, informal combination --
    the small-class vision pipeline wired up only as a cheap, reliable grounding source for ACT's
    own benefit -- and must not be reported as either arm later.

    Subclasses `HybridQwenActPolicy` purely to reuse its `act()`/`reset()`/`close()`/
    `replan_records` unchanged: none of them reference anything Qwen/`VLMBackend`-specific, only
    `self._grounder` (a plain, grounder-policy-agnostic `_Grounder` after this module's 2026-10-13
    refactor). Overriding `__init__` to build the `_Grounder` around a `SmallCnnGroundingPolicy`
    instead of a `MinimalVLMPolicy`, then delegating to the shared `_init_common()` helper, is the
    entire change needed. `grounding_policy=None` (e.g. for a CPU smoke test with no ONNX model on
    disk) matches `HybridQwenActPolicy(grounding_backend=None)`'s own convention: colour commands
    then correctly produce `has_target=False` (`debug["grounding_unavailable"] = "no
    grounding_policy configured"`), never a fabricated target."""

    def __init__(
        self,
        fast_layer: FastLayer,
        orchestrator: Optional[Union[StateMachineOrchestrator, LanguageOrchestrator]] = None,
        grounding_policy: Optional[Policy] = None,
        control_hz: float = CONTROL_HZ,
        margin_ms: float = DEFAULT_MARGIN_MS,
        reground_interval_s: float = DEFAULT_REGROUND_INTERVAL_S,
    ) -> None:
        grounder: Optional[_Grounder] = _Grounder(grounding_policy) if grounding_policy is not None else None
        self._init_common(fast_layer, orchestrator, grounder, control_hz, margin_ms, reground_interval_s)


class SmolVLADirectPolicy(Policy):
    """SMOLVLA_DIRECT (fast layer only) behind `Policy`. See module docstring for the full
    contract, including why `target_x_mm`/`target_y_mm` are NaN here."""

    def __init__(
        self,
        fast_layer: FastLayer,
        orchestrator: Optional[Union[StateMachineOrchestrator, LanguageOrchestrator]] = None,
        control_hz: float = CONTROL_HZ,
        margin_ms: float = DEFAULT_MARGIN_MS,
    ) -> None:
        # Advisory/debug-only slot: SmolVLA grounds itself on the instruction text and does not
        # need a resolved target to operate, but the slot is accepted (and exercised, into
        # debug) so a future real LanguageOrchestrator can start feeding this policy a
        # canonicalised instruction/target without a class rewrite -- see module docstring.
        self._orchestrator: Optional[Union[StateMachineOrchestrator, LanguageOrchestrator]] = orchestrator
        self._fast: _FastWorker = _FastWorker(fast_layer, control_hz, margin_ms)
        self.weights_status: str = fast_layer.weights_status
        self.trained_for_task: bool = _is_trained_for_task(fast_layer.weights_status)
        self._instruction_text: Optional[str] = None  # command_to_text() form, persists across calls
        self.last_debug: dict = {}

    @property
    def replan_records(self) -> list[ReplanRecord]:
        return self._fast.scheduler.records

    @property
    def worker_errors(self) -> list[str]:
        """See `HybridQwenActPolicy.worker_errors` -- same contract, no grounder here."""
        return list(self._fast.errors)

    def reset(self) -> None:
        if self._orchestrator is not None and hasattr(self._orchestrator, "reset"):
            self._orchestrator.reset()  # type: ignore[union-attr]
        self._instruction_text = None
        self.last_debug = {}

    def close(self) -> None:
        """Stops the background fast-layer worker. Not part of the `Policy` protocol."""
        self._fast.close()

    def act(self, image: np.ndarray, instruction: Optional[str], state: dict) -> PolicyCommand:
        now = time.monotonic()
        touch_mm = state.get("touch_mm")
        debug: dict[str, Any] = {
            "fast_layer": self._fast.fast_layer.name,
            "weights_status": self.weights_status,
            "trained_for_task": self.trained_for_task,
        }

        if self._orchestrator is not None and instruction is not None and touch_mm is not None:
            req, not_impl = _resolve_target_request(self._orchestrator, instruction, image, touch_mm)
            if not_impl is not None:
                debug["orchestrator_not_implemented"] = not_impl
            elif req is not None:
                debug["orchestrator_target_request"] = dataclasses.asdict(req)

        self._fast.poll_landed(now)  # drain; SmolVLA_DIRECT has no target staleness bookkeeping to update

        if instruction is not None:
            self._instruction_text = command_to_text(instruction)

        touch_finite = touch_mm is not None and bool(np.all(np.isfinite(touch_mm)))
        if self._instruction_text is not None and touch_finite:
            # (0.0, 0.0) placeholder: SmolVLAFastLayer.plan() never reads target_mm (see module
            # docstring) -- same convention run_experiment.py's run_live() uses for this option.
            submitted = self._fast.submit_if_due(
                now, image, (0.0, 0.0), (float(touch_mm[0]), float(touch_mm[1])),
                self._instruction_text, None, False, None,
            )
            debug["replan_submitted"] = submitted

        angles = self._fast.current_angles(now)
        debug["n_rejected_chunks"] = self._fast.n_rejected
        debug["last_rejection_reason"] = self._fast.last_rejection_reason
        self.last_debug = debug

        return PolicyCommand(
            target_x_mm=float("nan"), target_y_mm=float("nan"), angle_targets_deg=angles, debug=debug
        )
