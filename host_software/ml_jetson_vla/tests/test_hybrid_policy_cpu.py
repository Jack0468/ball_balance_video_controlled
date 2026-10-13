"""CPU-only checks for `core/hybrid_policy.py` (2026-10-07).

Covers: both `Policy` wrappers are constructible with `StubFastLayer` (no real weights, no
Jetson, no serial port); their `act()`/`reset()` match `core.policy_interface.Policy`'s exact
signatures (no `@runtime_checkable` on that `Protocol`, so this checks structurally rather than
via `isinstance` -- there is no pre-existing `JetsonExpertPolicy` protocol test to match the style
of; this is a new, minimal check style, documented as such); the safety gate actually rejects an
out-of-range stub chunk (never surfaced as `angle_targets_deg`); a short real replay against one
logged Track 4 session; and that importing `core.hybrid_policy` does not pull in torch or
lerobot, same style as `test_experiments_cpu.py`'s `ImportIsolationTests`.

Run (stdlib unittest, no new dependency):
    C:/Users/Admin/.conda/envs/ball_balance_env/python.exe host_software/ml_jetson_vla/tests/test_hybrid_policy_cpu.py
"""

from __future__ import annotations

import inspect
import os
import subprocess
import sys
import time
import unittest
from typing import Any, Optional
from unittest import mock

import numpy as np

_THIS_DIR: str = os.path.dirname(os.path.abspath(__file__))
_ML_JETSON_VLA_DIR: str = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_HOST_SOFTWARE_DIR: str = os.path.abspath(os.path.join(_ML_JETSON_VLA_DIR, ".."))
if _HOST_SOFTWARE_DIR not in sys.path:
    sys.path.append(_HOST_SOFTWARE_DIR)

from ml_jetson_vla.core.hybrid_policy import (  # noqa: E402
    HybridQwenActPolicy,
    HybridSmallCnnActPolicy,
    SmolVLADirectPolicy,
    _FastWorker,
)
from ml_jetson_vla.core.policy_interface import Policy, PolicyCommand  # noqa: E402
from ml_jetson_vla.deployment.bench_hybrid_qwen_act import TargetUpdate  # noqa: E402
from ml_jetson_vla.experiments.fast_layers import StubFastLayer  # noqa: E402
from ml_jetson_vla.experiments.run_experiment import VideoRowReader, load_session_telemetry  # noqa: E402
from ml_jetson_vla.experiments.schedule import CONTROL_HZ  # noqa: E402

_SESSION: str = os.path.join(
    _HOST_SOFTWARE_DIR, "data", "01_bronze", "session_jetson_track4_20260915_151627"
)
_FRAME: np.ndarray = np.zeros((480, 640, 3), dtype=np.uint8)


def _assert_satisfies_policy_protocol(case: unittest.TestCase, policy: Any) -> None:
    """Structural check: `Policy` (policy_interface.py) is a plain `typing.Protocol`, not
    `@runtime_checkable`, so `isinstance(policy, Policy)` would raise `TypeError` regardless of
    whether `policy`'s class explicitly subclasses it -- checked instead by matching `act()`'s/
    `reset()`'s real signatures against the protocol's documented ones."""
    # `Policy` is not @runtime_checkable (confirmed: isinstance()/issubclass() against it raises
    # TypeError even for a class that explicitly subclasses it), so membership is checked via the
    # MRO instead, plus the structural signature checks below.
    case.assertIn(Policy, type(policy).__mro__)
    case.assertTrue(callable(policy.act))
    case.assertTrue(callable(policy.reset))
    act_params = list(inspect.signature(policy.act).parameters)
    case.assertEqual(act_params, ["image", "instruction", "state"])
    case.assertEqual(list(inspect.signature(policy.reset).parameters), [])


def _wait_for(predicate: Any, timeout_s: float = 2.0, interval_s: float = 0.02) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval_s)
    return predicate()


def _wait_for_landed(
    worker: _FastWorker, timeout_s: float = 2.0, interval_s: float = 0.02
) -> list[Any]:
    """Polls `poll_landed()` until it returns a non-empty list or the timeout elapses."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        landed = worker.poll_landed(time.monotonic())
        if landed:
            return landed
        time.sleep(interval_s)
    return worker.poll_landed(time.monotonic())


class _FakeStore:
    """Duck-typed stand-in for `_Grounder.store` (a real `TargetStore`) -- lets tests control
    exactly what `latest()` returns on each call without running a real grounding thread."""

    def __init__(self, update: TargetUpdate) -> None:
        self._update = update
        self.slow_calls: int = 1
        self.parse_failures: int = 0

    def latest(self) -> TargetUpdate:
        return self._update


class _FakeGrounder:
    """Duck-typed stand-in for `_Grounder` -- same shape `HybridQwenActPolicy.act()` reads
    (`.store.latest()`, `.store.slow_calls`, `.store.parse_failures`, `.request()`, `.close()`),
    with no real background thread, so tests can deterministically control the "slow loop"
    target a policy sees without racing real wall-clock time."""

    def __init__(self, done_monotonic_s: float, x_mm: float = 10.0, y_mm: float = 20.0) -> None:
        self.store: _FakeStore = _FakeStore(TargetUpdate(1, x_mm, y_mm, done_monotonic_s, "go_red", 0.01))

    def request(self, frame: np.ndarray, instruction: str) -> bool:
        return True

    def close(self) -> None:
        pass


class _FakeSmallCnnGroundingPolicy:
    """Duck-typed stand-in for `core.small_cnn_grounder.SmallCnnGroundingPolicy` -- a trivial,
    near-instant `Policy`-shaped grounding implementation (same contract `_Grounder` requires:
    `.act(frame, instruction, state) -> PolicyCommand` plus a `.last_debug` dict carrying
    `parse_ok`), so `HybridSmallCnnActPolicyTests` below exercise the REAL `_Grounder`/
    `_FastWorker` background threads end to end without needing onnxruntime/cv2/a real ONNX
    model file. Counts calls so tests can assert whether grounding was actually requested."""

    def __init__(self, x_mm: float = 10.0, y_mm: float = 20.0) -> None:
        self.n_calls: int = 0
        self.last_debug: dict = {}
        self._x_mm = x_mm
        self._y_mm = y_mm

    def act(self, image: np.ndarray, instruction: Optional[str], state: dict) -> PolicyCommand:
        self.n_calls += 1
        self.last_debug = {"parse_ok": True, "instruction": instruction}
        return PolicyCommand(target_x_mm=self._x_mm, target_y_mm=self._y_mm)

    def reset(self) -> None:
        self.last_debug = {}


class ConstructionAndProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self._opened: list[Any] = []

    def tearDown(self) -> None:
        for p in self._opened:
            p.close()

    def _hybrid(self, **kwargs: Any) -> HybridQwenActPolicy:
        p = HybridQwenActPolicy(StubFastLayer(value_deg=0.0, chunk_len=5), **kwargs)
        self._opened.append(p)
        return p

    def _smolvla(self, **kwargs: Any) -> SmolVLADirectPolicy:
        p = SmolVLADirectPolicy(StubFastLayer(value_deg=0.0, chunk_len=5), **kwargs)
        self._opened.append(p)
        return p

    def test_hybrid_policy_constructs_with_stub_and_no_grounding_backend(self) -> None:
        p = self._hybrid()
        _assert_satisfies_policy_protocol(self, p)
        self.assertFalse(p.trained_for_task)
        self.assertIn("STUB", p.weights_status)

    def test_smolvla_direct_constructs_with_stub(self) -> None:
        p = self._smolvla()
        _assert_satisfies_policy_protocol(self, p)
        self.assertFalse(p.trained_for_task)

    def test_act_without_touch_mm_returns_a_command_without_crashing(self) -> None:
        p = self._hybrid()
        cmd = p.act(_FRAME, "hold", {})
        self.assertIsInstance(cmd, PolicyCommand)
        self.assertIsNone(cmd.angle_targets_deg)
        self.assertTrue(np.isnan(cmd.target_x_mm))

    def test_smolvla_direct_target_is_always_nan_by_design(self) -> None:
        p = self._smolvla()
        cmd = p.act(_FRAME, "go green", {"touch_mm": (1.0, 2.0)})
        self.assertTrue(np.isnan(cmd.target_x_mm))
        self.assertTrue(np.isnan(cmd.target_y_mm))

    def test_hybrid_accepts_language_orchestrator_slot(self) -> None:
        from ml_jetson_vla.experiments.orchestrators import LanguageOrchestrator

        p = self._hybrid(orchestrator=LanguageOrchestrator())  # unconfigured stub
        cmd = p.act(_FRAME, "bring me the green one", {"touch_mm": (0.0, 0.0)})
        self.assertIn("orchestrator_not_implemented", p.last_debug)
        self.assertIsInstance(cmd, PolicyCommand)


class SafetyGateTests(unittest.TestCase):
    def test_hybrid_rejects_out_of_range_stub_chunk(self) -> None:
        policy = HybridQwenActPolicy(StubFastLayer(value_deg=12.0, chunk_len=5))  # > 11.025 deg limit
        try:
            touch = (0.0, 0.0)
            for _ in range(5):
                cmd = policy.act(_FRAME, "hold", {"touch_mm": touch})
                time.sleep(0.03)
            _wait_for(lambda: policy.last_debug.get("n_rejected_chunks", 0) > 0, timeout_s=2.0)
            for _ in range(5):
                cmd = policy.act(_FRAME, None, {"touch_mm": touch})
                time.sleep(0.03)
            self.assertGreater(policy.last_debug["n_rejected_chunks"], 0)
            self.assertIsNotNone(policy.last_debug["last_rejection_reason"])
            self.assertIn("outside", policy.last_debug["last_rejection_reason"])
            self.assertIsNone(cmd.angle_targets_deg)  # never surfaced, whole chunk rejected
        finally:
            policy.close()

    def test_smolvla_direct_rejects_out_of_range_stub_chunk(self) -> None:
        policy = SmolVLADirectPolicy(StubFastLayer(value_deg=99.0, chunk_len=5))
        try:
            touch = (0.0, 0.0)
            for _ in range(5):
                cmd = policy.act(_FRAME, "go green", {"touch_mm": touch})
                time.sleep(0.03)
            _wait_for(lambda: policy.last_debug.get("n_rejected_chunks", 0) > 0, timeout_s=2.0)
            self.assertGreater(policy.last_debug["n_rejected_chunks"], 0)
            self.assertIsNone(cmd.angle_targets_deg)
        finally:
            policy.close()

    def test_hybrid_accepts_in_range_stub_chunk(self) -> None:
        policy = HybridQwenActPolicy(StubFastLayer(value_deg=1.0, chunk_len=5))
        try:
            touch = (0.0, 0.0)
            cmd = None
            for _ in range(10):
                cmd = policy.act(_FRAME, "hold", {"touch_mm": touch})
                time.sleep(0.03)
                if cmd.angle_targets_deg is not None:
                    break
            self.assertIsNotNone(cmd.angle_targets_deg)
            self.assertEqual(cmd.angle_targets_deg, (1.0, 1.0, 1.0))
            self.assertEqual(policy.last_debug["n_rejected_chunks"], 0)
        finally:
            policy.close()


class HybridSmallCnnActPolicyTests(unittest.TestCase):
    """`HybridSmallCnnActPolicy` (2026-10-13, debugging/staging config -- see its docstring):
    grounds colour commands with a small-CNN pipeline instead of Qwen. These tests drive it with
    `_FakeSmallCnnGroundingPolicy` (a trivial, near-instant stand-in -- no onnxruntime/cv2/real
    ONNX model needed) through the REAL `_Grounder`/`_FastWorker` background threads, the same
    style `ConstructionAndProtocolTests`/`SafetyGateTests` above already use for
    `HybridQwenActPolicy`."""

    def setUp(self) -> None:
        self._opened: list[Any] = []

    def tearDown(self) -> None:
        for p in self._opened:
            p.close()

    def _policy(self, grounding_policy: Any = None, **kwargs: Any) -> HybridSmallCnnActPolicy:
        p = HybridSmallCnnActPolicy(
            StubFastLayer(value_deg=0.0, chunk_len=5), grounding_policy=grounding_policy, **kwargs
        )
        self._opened.append(p)
        return p

    def test_constructs_and_satisfies_policy_protocol(self) -> None:
        p = self._policy()
        _assert_satisfies_policy_protocol(self, p)
        self.assertFalse(p.trained_for_task)
        self.assertIn("STUB", p.weights_status)

    def test_no_grounding_policy_configured_produces_no_target_for_colour_command(self) -> None:
        """Matches `HybridQwenActPolicy(grounding_backend=None)`'s own convention: omitting the
        grounding policy entirely must degrade gracefully, never crash or fabricate a target."""
        p = self._policy()
        cmd = p.act(_FRAME, "go_red", {"touch_mm": (0.0, 0.0)})
        self.assertEqual(p.last_debug.get("grounding_unavailable"), "no grounding_policy configured")
        self.assertTrue(np.isnan(cmd.target_x_mm))

    def test_colour_command_triggers_a_grounding_call(self) -> None:
        fake = _FakeSmallCnnGroundingPolicy()
        p = self._policy(grounding_policy=fake)
        cmd = p.act(_FRAME, "go_red", {"touch_mm": (0.0, 0.0)})
        self.assertTrue(p.last_debug.get("grounding_requested"))
        self.assertTrue(_wait_for(lambda: fake.n_calls > 0))
        self.assertEqual(fake.n_calls, 1)
        self.assertIsInstance(cmd, PolicyCommand)

    def test_directional_command_does_not_trigger_a_grounding_call(self) -> None:
        fake = _FakeSmallCnnGroundingPolicy()
        p = self._policy(grounding_policy=fake)
        for _ in range(5):
            p.act(_FRAME, "hold", {"touch_mm": (0.0, 0.0)})
            time.sleep(0.03)
        self.assertEqual(fake.n_calls, 0)
        self.assertNotIn("grounding_requested", p.last_debug)

    def test_grounding_call_does_not_block_act(self) -> None:
        """`_Grounder.request()`/`store.latest()` are both non-blocking by design -- a slow
        grounding implementation must never make `act()` itself block waiting on it."""

        class _SlowGroundingPolicy(_FakeSmallCnnGroundingPolicy):
            def act(self, image: np.ndarray, instruction: Optional[str], state: dict) -> PolicyCommand:
                time.sleep(0.5)
                return super().act(image, instruction, state)

        fake = _SlowGroundingPolicy()
        p = self._policy(grounding_policy=fake)
        t0 = time.monotonic()
        cmd = p.act(_FRAME, "go_red", {"touch_mm": (0.0, 0.0)})
        elapsed = time.monotonic() - t0
        self.assertLess(elapsed, 0.3)  # returns long before the fake's 0.5s "inference" finishes
        self.assertTrue(np.isnan(cmd.target_x_mm))  # nothing has landed yet -- degrades gracefully

    def test_degrades_gracefully_before_first_real_detection_lands(self) -> None:
        fake = _FakeSmallCnnGroundingPolicy(x_mm=10.0, y_mm=20.0)
        p = self._policy(grounding_policy=fake)

        cmd = p.act(_FRAME, "go_red", {"touch_mm": (0.0, 0.0)})
        self.assertFalse(p.last_debug["has_target"])  # grounding just requested, nothing landed yet
        self.assertTrue(np.isnan(cmd.target_x_mm))
        self.assertTrue(np.isnan(cmd.target_y_mm))

        self.assertTrue(_wait_for(lambda: fake.n_calls > 0))
        time.sleep(0.1)  # give _Grounder's background thread time to store.put() the result
        cmd2 = p.act(_FRAME, None, {"touch_mm": (0.0, 0.0)})
        self.assertTrue(p.last_debug["has_target"])
        self.assertEqual(cmd2.target_x_mm, 10.0)
        self.assertEqual(cmd2.target_y_mm, 20.0)


class PeriodicRegroundingTests(unittest.TestCase):
    """2026-10-13 fix: a held colour command must be periodically re-submitted to the grounder,
    not grounded once and then left stale for as long as the command is held (flagged
    2026-10-06; `EXPERIMENT_OPTIONS_PLAN_2026_10_07.md` section 4 "Known limit #2"; target-age
    p90 measured 4.2-8.9s across real sessions with no re-grounding). Driven through the REAL
    `_Grounder` background thread with `_FakeSmallCnnGroundingPolicy` (near-instant), same style
    as `HybridSmallCnnActPolicyTests` above, so this exercises the actual cadence/timing logic in
    `act()` rather than a mocked stand-in."""

    def setUp(self) -> None:
        self._opened: list[Any] = []

    def tearDown(self) -> None:
        for p in self._opened:
            p.close()

    def _policy(self, grounding_policy: Any, **kwargs: Any) -> HybridSmallCnnActPolicy:
        p = HybridSmallCnnActPolicy(
            StubFastLayer(value_deg=0.0, chunk_len=5), grounding_policy=grounding_policy, **kwargs
        )
        self._opened.append(p)
        return p

    def test_periodic_reground_keeps_target_age_bounded_while_command_is_held(self) -> None:
        fake = _FakeSmallCnnGroundingPolicy()
        p = self._policy(fake, reground_interval_s=0.1)
        touch = (0.0, 0.0)

        # First tick resolves "go_red" and issues the initial grounding request.
        p.act(_FRAME, "go_red", {"touch_mm": touch})
        self.assertTrue(_wait_for(lambda: fake.n_calls > 0))

        # Hold the SAME instruction (None passed every subsequent tick, matching real usage --
        # the audio command is only re-delivered on change) across several re-grounding cycles.
        max_age_ms = 0.0
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            p.act(_FRAME, None, {"touch_mm": touch})
            if p.last_debug.get("has_target"):
                latest = p._grounder.store.latest()  # type: ignore[union-attr]
                age_ms = max(0.0, time.monotonic() - latest.done_monotonic_s) * 1000.0
                max_age_ms = max(max_age_ms, age_ms)
            time.sleep(1.0 / CONTROL_HZ)

        # The whole point of the fix: more than one grounding call happened while the SAME
        # instruction was held, not just the single initial one.
        self.assertGreater(fake.n_calls, 1)
        # And the measured target age never grew anywhere near the old 4.2-8.9s failure mode --
        # bounded by the project's own proposed <=2000ms criterion (section 3 of the plan doc).
        self.assertLess(max_age_ms, 2000.0)

    def test_directional_command_clears_active_grounding_and_stops_periodic_requests(self) -> None:
        fake = _FakeSmallCnnGroundingPolicy()
        p = self._policy(fake, reground_interval_s=0.05)
        touch = (0.0, 0.0)

        p.act(_FRAME, "go_red", {"touch_mm": touch})
        self.assertTrue(_wait_for(lambda: fake.n_calls > 0))
        self.assertIsNotNone(p._active_grounding_instruction)

        # A directional/hold command supersedes the held colour target.
        p.act(_FRAME, "hold", {"touch_mm": touch})
        self.assertIsNone(p._active_grounding_instruction)

        n_calls_at_switch = fake.n_calls
        for _ in range(10):
            p.act(_FRAME, None, {"touch_mm": touch})
            time.sleep(0.05)
        # No further periodic re-grounding once the colour command is no longer active.
        self.assertEqual(fake.n_calls, n_calls_at_switch)

    def test_smolvla_direct_unaffected_by_periodic_reground_change(self) -> None:
        """`SmolVLADirectPolicy` has no `_Grounder`/`_active_grounding_instruction` at all --
        confirms the fix did not leak into its construction or `act()` behaviour."""
        p = SmolVLADirectPolicy(StubFastLayer(value_deg=0.0, chunk_len=5))
        try:
            self.assertFalse(hasattr(p, "_active_grounding_instruction"))
            self.assertFalse(hasattr(p, "_grounder"))
            cmd = p.act(_FRAME, "go green", {"touch_mm": (1.0, 2.0)})
            self.assertTrue(np.isnan(cmd.target_x_mm))
            self.assertTrue(np.isnan(cmd.target_y_mm))
        finally:
            p.close()


@unittest.skipUnless(os.path.exists(_SESSION), f"real Track 4 session not found: {_SESSION}")
class RealSessionReplayTests(unittest.TestCase):
    """Drives HybridQwenActPolicy (StubFastLayer, no grounding backend) against one real logged
    Track 4 session's frames/telemetry, read-only (host_software/data/ is not written to)."""

    def test_replay_short_session_with_stub(self) -> None:
        tel = load_session_telemetry(_SESSION)
        # This session's first ~20s is all colour commands (no grounding_backend here -> no
        # target); 25s crosses into the first directional command ("forward" at t=20.08s per a
        # one-off audit of this session's audio_command column), so this window exercises both
        # the "no target yet" path and the fixed-target/replan/angle-output path for real.
        max_sim_seconds = 25.0
        tick_times = np.arange(0.0, max_sim_seconds, 1.0 / CONTROL_HZ)
        tick_rows = np.searchsorted(tel.t_s, tick_times, side="right") - 1

        policy = HybridQwenActPolicy(StubFastLayer(value_deg=0.5, chunk_len=10))
        reader = VideoRowReader(os.path.join(_SESSION, "rgb_video.mp4"))
        n_ticks = 0
        n_with_target = 0
        n_with_angles = 0
        last_cmd = ""
        try:
            for row_i in tick_rows:
                if row_i < 0:
                    continue
                row = int(row_i)
                n_ticks += 1
                cmd = tel.commands[row]
                instruction: Optional[str] = None
                if cmd and cmd != last_cmd:
                    last_cmd = cmd
                    instruction = cmd  # raw vocabulary form -- Policy.act()'s documented convention
                frame = reader.read(row)
                touch_mm = (float(tel.touch[row, 0]), float(tel.touch[row, 1]))
                command = policy.act(frame, instruction, {"touch_mm": touch_mm})
                if bool(policy.last_debug.get("has_target", False)):
                    n_with_target += 1
                if command.angle_targets_deg is not None:
                    n_with_angles += 1
                time.sleep(1.0 / CONTROL_HZ / 4.0)  # real worker thread -- give it a chance to run
        finally:
            reader.close()
            policy.close()

        print(
            f"[replay] session={os.path.basename(_SESSION)} n_ticks={n_ticks} "
            f"n_with_target={n_with_target} n_with_angle_output={n_with_angles} "
            f"n_rejected={policy.last_debug.get('n_rejected_chunks')} "
            f"n_replans_landed={len(policy.replan_records)}"
        )
        self.assertGreater(n_ticks, 0)
        self.assertEqual(policy.last_debug.get("n_rejected_chunks"), 0)  # 0.5deg stub is in range
        # Colour commands (this session's first ~20s) correctly produce no target -- no
        # grounding_backend configured, so has_target stays False for those ticks. The
        # directional command after that gives a fixed target with no grounding needed, so this
        # window should show both a real target and real accepted chunks landing.
        self.assertGreater(n_with_target, 0)
        self.assertGreater(n_with_angles, 0)
        self.assertGreater(len(policy.replan_records), 0)


class CrossThreadRaceAndStaleTargetTests(unittest.TestCase):
    """Deterministic reproductions for findings 1 and 2 (2026-10-07 review) -- both are real
    concurrency bugs, so each is driven via controlled, explicit timestamps / a fake grounder
    store rather than relying on real thread-timing to hit the race window."""

    def test_finding1_act_does_not_raise_when_target_done_s_is_after_a_stale_now(self) -> None:
        """Finding 1: the grounder's background thread stamps `done_monotonic_s` with its OWN
        later `time.monotonic()` call, independent of the `now` `act()` already captured at the
        top of the method -- so `target_done_s` can legitimately be later than that `now`.
        Reproduced directly: inject a target whose `done_monotonic_s` is set ahead of a `now`
        `act()`'s first `time.monotonic()` call is forced to return (mocked), while leaving the
        second, later-in-the-method call (the fix's re-sample point) real."""
        policy = HybridQwenActPolicy(StubFastLayer(value_deg=0.5, chunk_len=5))
        try:
            real_now = time.monotonic()
            future_done_s = real_now + 5.0  # "later" than the stale `now` captured below
            policy._grounder = _FakeGrounder(future_done_s)  # type: ignore[assignment]
            stale_now = real_now  # the `now` act() would have captured "before" the race

            call_count = {"n": 0}
            real_monotonic = time.monotonic

            def fake_monotonic() -> float:
                call_count["n"] += 1
                # 1st call = act()'s `now = time.monotonic()`; every later call (the finding-1
                # fix's re-sample at the point of use) gets the real, unmocked clock.
                return stale_now if call_count["n"] == 1 else real_monotonic()

            with mock.patch("ml_jetson_vla.core.hybrid_policy.time.monotonic", side_effect=fake_monotonic):
                try:
                    cmd = policy.act(_FRAME, None, {"touch_mm": (0.0, 0.0)})
                except ValueError as exc:
                    self.fail(f"act() raised on a cross-thread skew it must tolerate: {exc}")
            self.assertIsInstance(cmd, PolicyCommand)
            self.assertTrue(policy.last_debug["has_target"])
        finally:
            policy.close()

    def test_finding2_fast_worker_reports_the_target_captured_at_submission_not_at_landing(self) -> None:
        """Finding 2, mechanism-level: `_FastWorker.poll_landed()` must return the `target_done_s`
        that was actually passed to `submit_if_due()` for that specific chunk, carried through the
        in/out queues -- not something read fresh at landing time (there is nothing else it could
        be contaminated by here, since this test drives the worker directly with one fixed value)."""
        worker = _FastWorker(StubFastLayer(value_deg=0.5, chunk_len=3))
        try:
            submitted_target_done_s = 111.0
            submitted = worker.submit_if_due(
                now=0.0,
                frame=_FRAME,
                target_mm=(1.0, 2.0),
                state_mm=(0.0, 0.0),
                instruction="hold",
                age=0.0,
                new_arrived=True,
                target_done_s=submitted_target_done_s,
            )
            self.assertTrue(submitted)
            landed = _wait_for_landed(worker)
            self.assertEqual(len(landed), 1)
            record, reported_target_done_s = landed[0]
            self.assertTrue(record.accepted)
            self.assertEqual(reported_target_done_s, submitted_target_done_s)
        finally:
            worker.close()

    def test_finding2_used_done_s_does_not_jump_to_a_newer_unused_target(self) -> None:
        """Finding 2, policy-level: a newer grounder target (t2) lands in the store between a
        chunk's submission (against t1) and that chunk being polled as landed. Before the fix,
        `_used_done_s` was set from the store's CURRENT target at landing time (t2), wrongly
        marking t2 as already used even though no inference ever ran against it."""
        policy = HybridQwenActPolicy(StubFastLayer(value_deg=0.5, chunk_len=5))
        try:
            base = time.monotonic()
            t1 = base - 2.0  # the target actually submitted with the first chunk
            t2 = base - 1.0  # newer than t1, arrives in the store before that chunk lands
            touch = (0.0, 0.0)

            policy._grounder = _FakeGrounder(t1)  # type: ignore[assignment]
            cmd1 = policy.act(_FRAME, None, {"touch_mm": touch})
            self.assertTrue(policy.last_debug["replan_submitted"])  # first call always submits

            # Give the real background worker thread time to finish the (near-instant) stub
            # inference and queue its result -- without yet draining it via another act() call.
            self.assertTrue(_wait_for(lambda: policy._fast._out_q.qsize() > 0, timeout_s=2.0))

            # Simulate the race: a newer grounding result (t2) now supersedes t1 in the store,
            # before the policy has polled the chunk that was submitted against t1.
            policy._grounder = _FakeGrounder(t2)  # type: ignore[assignment]

            cmd2 = policy.act(_FRAME, None, {"touch_mm": touch})  # drains the t1 chunk; sees t2 as current

            self.assertEqual(policy._used_done_s, t1)  # the chunk actually used t1, not t2
            self.assertGreater(t2, policy._used_done_s)  # t2 is still free to be treated as new
            self.assertIsInstance(cmd1, PolicyCommand)
            self.assertIsInstance(cmd2, PolicyCommand)
        finally:
            policy.close()


class ImportIsolationTests(unittest.TestCase):
    def test_package_import_does_not_pull_torch_or_lerobot(self) -> None:
        code = (
            "import sys\n"
            "import ml_jetson_vla.core.hybrid_policy\n"
            "bad = [m for m in ('torch', 'lerobot') if m in sys.modules]\n"
            "print('BAD=' + ','.join(bad))\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code], cwd=_HOST_SOFTWARE_DIR, capture_output=True, text=True, timeout=120
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), "BAD=", proc.stdout)


if __name__ == "__main__":
    unittest.main()
