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

import numpy as np

_THIS_DIR: str = os.path.dirname(os.path.abspath(__file__))
_ML_JETSON_VLA_DIR: str = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_HOST_SOFTWARE_DIR: str = os.path.abspath(os.path.join(_ML_JETSON_VLA_DIR, ".."))
if _HOST_SOFTWARE_DIR not in sys.path:
    sys.path.append(_HOST_SOFTWARE_DIR)

from ml_jetson_vla.core.hybrid_policy import HybridQwenActPolicy, SmolVLADirectPolicy  # noqa: E402
from ml_jetson_vla.core.policy_interface import Policy, PolicyCommand  # noqa: E402
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
