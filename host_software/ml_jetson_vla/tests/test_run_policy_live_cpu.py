"""CPU-only checks for `experiments/run_policy_live.py` (2026-10-13).

This is the live (real-serial-sends) sibling of `run_policy_replay.py`, built so
`HybridQwenActPolicy`/`HybridSmallCnnActPolicy`/`SmolVLADirectPolicy` can all be tested on real
hardware through the `Policy` abstraction, not just in replay -- closing the gap where
`run_experiment.py`'s Qwen-specific `run_live()` was the only live driver that existed.

No real camera/serial/Jetson here: `cv2.VideoCapture`/`serial.Serial` are monkeypatched at their
own module attribute (`cv2.VideoCapture`, `serial.Serial` -- `run_policy_live.run_live()` imports
both lazily inside the function body, so patching the class on the already-imported module object
is enough). `build_policy()` itself is monkeypatched at `run_policy_live`'s own import site, same
style `test_run_policy_replay_cpu.py`/`test_hybrid_policy_cpu.py` use for monkeypatching other
construction-site classes -- a trivial `_FakePolicy` stands in, with a controllable `act()`, so
none of these tests ever construct a real `HybridQwenActPolicy`/background thread.

Covers exactly the behaviours this driver must get right (per the task): the motor-consent gate
actually refuses without `--i-understand-this-drives-motors` (mirroring `run_experiment.py`'s own
refusal message/behaviour and writing no output file); a tick with `angle_targets_deg is None`
sends nothing; a tick with a real value sends the correctly `serial_protocol.format_angle_line()`
-formatted line; `LEVEL_LINE` is sent on both clean exit and exception; and the telemetry CSV has
every column `evaluate_system_control.REQUIRED_COLUMNS` needs.

Run (stdlib unittest, no new dependency):
    C:/Users/Admin/.conda/envs/ball_balance_env/python.exe host_software/ml_jetson_vla/tests/test_run_policy_live_cpu.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
import time
import unittest
from typing import Any, Callable, Optional
from unittest import mock

import numpy as np

_THIS_DIR: str = os.path.dirname(os.path.abspath(__file__))
_ML_JETSON_VLA_DIR: str = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_HOST_SOFTWARE_DIR: str = os.path.abspath(os.path.join(_ML_JETSON_VLA_DIR, ".."))
_REPO_ROOT_DIR: str = os.path.abspath(os.path.join(_HOST_SOFTWARE_DIR, ".."))
for _p in (_HOST_SOFTWARE_DIR, _REPO_ROOT_DIR):
    if _p not in sys.path:
        sys.path.append(_p)
_EVALUATIONS_DIR: str = os.path.join(_HOST_SOFTWARE_DIR, "evaluations")
if _EVALUATIONS_DIR not in sys.path:
    sys.path.append(_EVALUATIONS_DIR)

from ml_jetson_vla.core.policy_interface import PolicyCommand  # noqa: E402
from ml_jetson_vla.experiments import run_policy_live as rpl  # noqa: E402
from ml_jetson_vla.experiments.run_experiment import EXIT_USAGE, MOTOR_GATE_FLAG  # noqa: E402
from ml_jetson_vla.experiments.serial_protocol import LEVEL_LINE  # noqa: E402

from evaluate_system_control import REQUIRED_COLUMNS  # noqa: E402


class _FakePolicy:
    """Duck-typed stand-in for a `core.hybrid_policy.py` `Policy` -- `act_fn(tick_index)` decides
    what each call returns, so tests can deterministically control what a given tick sees without
    a real background thread/scheduler."""

    def __init__(self, act_fn: Callable[[int], PolicyCommand]) -> None:
        self._act_fn = act_fn
        self._n = 0
        self.weights_status = "STUB (no model), constant 0.0 deg"
        self.trained_for_task = False
        self.last_debug: dict = {}
        self.replan_records: list = []
        self.closed = False

    def act(self, image: Any, instruction: Optional[str], state: dict) -> PolicyCommand:
        command = self._act_fn(self._n)
        self._n += 1
        self.last_debug = command.debug or {}
        return command

    def reset(self) -> None:
        self.last_debug = {}

    def close(self) -> None:
        self.closed = True


class _FakeCamera:
    def __init__(self, index: int) -> None:
        self.index = index
        self.released = False

    def isOpened(self) -> bool:
        return True

    def read(self) -> tuple:
        return True, np.zeros((480, 640, 3), dtype=np.uint8)

    def release(self) -> None:
        self.released = True


class _NeverOpensCamera(_FakeCamera):
    def isOpened(self) -> bool:
        return False


class _FakeSerial:
    """Records every write; `readline()` returns a canned valid uplink line repeatedly (a small
    sleep approximates the real `timeout=0.1` pacing so the uplink thread doesn't busy-spin)."""

    instances: list = []

    def __init__(self, port: str, baud: int, timeout: float = 0.1) -> None:
        self.port = port
        self.baud = baud
        self.writes: list = []
        self.closed = False
        _FakeSerial.instances.append(self)

    def readline(self) -> bytes:
        time.sleep(0.01)
        return b"T,1,100,500,-300,1,0,0,0\n"

    def write(self, data: bytes) -> None:
        self.writes.append(data)

    def close(self) -> None:
        self.closed = True

    @property
    def decoded_writes(self) -> list:
        return [d.decode("ascii") for d in self.writes]


class _LivePolicyTestCase(unittest.TestCase):
    """Common fixture: a temp dir holding a real 3x3 homography .npy (loaded for real by
    `run_live()`, not mocked -- a bad path should still be caught at startup) and the output
    JSON path, cleaned up automatically."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.out_path = os.path.join(self._tmp.name, "out.json")
        self.homography_path = os.path.join(self._tmp.name, "homography.npy")
        np.save(self.homography_path, np.eye(3))
        _FakeSerial.instances = []

    def _argv(self, policy: str = "hybrid_qwen_act", max_run_s: float = 0.15,
              include_gate: bool = True) -> list:
        argv = [
            "--policy", policy,
            "--out", self.out_path,
            "--serial-port", "COM_FAKE",
            "--homography-npy", self.homography_path,
            "--instruction", "go_green",
            "--max-run-s", str(max_run_s),
        ]
        if include_gate:
            argv.append(MOTOR_GATE_FLAG)
        return argv

    def _run_main(self, act_fn: Callable[[int], PolicyCommand], **argv_kwargs: Any) -> int:
        fake_policy = _FakePolicy(act_fn)
        self.fake_policy = fake_policy
        with mock.patch.object(rpl, "build_policy", return_value=fake_policy), \
             mock.patch("cv2.VideoCapture", _FakeCamera), \
             mock.patch("serial.Serial", _FakeSerial):
            return rpl.main(self._argv(**argv_kwargs))

    def _load_result(self) -> dict:
        with open(self.out_path, encoding="utf-8") as fh:
            return json.load(fh)


class MotorGateTests(_LivePolicyTestCase):
    def test_refuses_without_gate_flag(self) -> None:
        code = rpl.main(self._argv(include_gate=False))
        self.assertEqual(code, EXIT_USAGE)
        self.assertFalse(os.path.exists(self.out_path))  # refused before run_live() ever wrote anything

    def test_runs_with_gate_flag(self) -> None:
        code = self._run_main(lambda n: PolicyCommand(target_x_mm=float("nan"), target_y_mm=float("nan")))
        self.assertEqual(code, 0)
        self.assertTrue(os.path.exists(self.out_path))
        self.assertTrue(self.fake_policy.closed)


class TickBehaviourTests(_LivePolicyTestCase):
    def test_tick_with_no_target_sends_nothing(self) -> None:
        code = self._run_main(
            lambda n: PolicyCommand(target_x_mm=float("nan"), target_y_mm=float("nan"), angle_targets_deg=None)
        )
        self.assertEqual(code, 0)
        self.assertEqual(len(_FakeSerial.instances), 1)
        ser = _FakeSerial.instances[0]
        # Only the LEVEL_LINE (sent once on exit) should ever have been written -- never an
        # "A,..." angle line, since every act() call returned angle_targets_deg=None.
        angle_lines = [w for w in ser.decoded_writes if w.startswith("A,") and w.strip() != LEVEL_LINE]
        self.assertEqual(angle_lines, [])
        self.assertIn(LEVEL_LINE + "\n", ser.decoded_writes)

    def test_tick_with_real_value_sends_correctly_formatted_line(self) -> None:
        code = self._run_main(
            lambda n: PolicyCommand(target_x_mm=10.0, target_y_mm=20.0, angle_targets_deg=(1.0, -2.5, 0.0))
        )
        self.assertEqual(code, 0)
        ser = _FakeSerial.instances[0]
        self.assertIn("A,1.0000,-2.5000,0.0000\n", ser.decoded_writes)
        result = self._load_result()
        self.assertGreater(result["lines_sent"], 0)

    def test_level_line_sent_on_clean_exit(self) -> None:
        self._run_main(lambda n: PolicyCommand(target_x_mm=float("nan"), target_y_mm=float("nan")))
        ser = _FakeSerial.instances[0]
        self.assertIn(LEVEL_LINE + "\n", ser.decoded_writes)

    def test_level_line_sent_on_exception(self) -> None:
        def _boom(n: int) -> PolicyCommand:
            if n >= 1:
                raise RuntimeError("simulated policy failure")
            return PolicyCommand(target_x_mm=float("nan"), target_y_mm=float("nan"))

        code = self._run_main(_boom)
        self.assertEqual(code, 1)  # EXIT_FAILED -- recorded, never a fabricated "ok" result
        ser = _FakeSerial.instances[0]
        self.assertIn(LEVEL_LINE + "\n", ser.decoded_writes)
        result = self._load_result()
        self.assertEqual(result["status"], "failed")
        self.assertIn("simulated policy failure", result["error"])

    def test_telemetry_csv_has_required_columns(self) -> None:
        self._run_main(
            lambda n: PolicyCommand(target_x_mm=1.0, target_y_mm=2.0, angle_targets_deg=(0.0, 0.0, 0.0))
        )
        telemetry_path = os.path.abspath(self.out_path) + ".telemetry.csv"
        self.assertTrue(os.path.exists(telemetry_path))
        with open(telemetry_path, newline="", encoding="utf-8") as fh:
            header = next(csv.reader(fh))
        for col in REQUIRED_COLUMNS:
            self.assertIn(col, header)

    def test_camera_open_failure_raises_and_is_recorded(self) -> None:
        fake_policy = _FakePolicy(lambda n: PolicyCommand(target_x_mm=float("nan"), target_y_mm=float("nan")))
        with mock.patch.object(rpl, "build_policy", return_value=fake_policy), \
             mock.patch("cv2.VideoCapture", _NeverOpensCamera), \
             mock.patch("serial.Serial", _FakeSerial):
            code = rpl.main(self._argv())
        self.assertEqual(code, 1)
        result = self._load_result()
        self.assertEqual(result["status"], "failed")
        self.assertIn("cannot open camera", result["error"])

    def test_bad_homography_shape_raises_and_is_recorded(self) -> None:
        np.save(self.homography_path, np.eye(2))  # wrong shape
        fake_policy = _FakePolicy(lambda n: PolicyCommand(target_x_mm=float("nan"), target_y_mm=float("nan")))
        with mock.patch.object(rpl, "build_policy", return_value=fake_policy), \
             mock.patch("cv2.VideoCapture", _FakeCamera), \
             mock.patch("serial.Serial", _FakeSerial):
            code = rpl.main(self._argv())
        self.assertEqual(code, 1)
        result = self._load_result()
        self.assertEqual(result["status"], "failed")
        self.assertIn("3x3", result["error"])


class ParseArgsTests(unittest.TestCase):
    def test_defaults_match_run_policy_replay_conventions(self) -> None:
        args = rpl.parse_args([
            "--policy", "hybrid_qwen_act", "--out", "x.json", "--serial-port", "COM_FAKE",
            "--homography-npy", "h.npy", "--instruction", "go_green",
        ])
        self.assertEqual(args.fast_layer, "stub")
        self.assertEqual(args.device, "cpu")
        self.assertTrue(args.small_cnn_grounding)
        self.assertEqual(args.baud, 2000000)
        self.assertEqual(args.max_run_s, 10.0)

    def test_missing_required_flag_exits(self) -> None:
        argv = ["--policy", "hybrid_qwen_act", "--out", "x.json"]  # missing serial/homography/instruction
        with self.assertRaises(SystemExit):
            rpl.parse_args(argv)


if __name__ == "__main__":
    unittest.main()
