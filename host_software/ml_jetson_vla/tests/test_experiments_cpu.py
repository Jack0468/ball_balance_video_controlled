"""CPU-only checks for host_software/ml_jetson_vla/experiments (2026-10-07).

Covers: the option-S command map, the option-L stub, the serial formatter and out-of-range
rejection, uplink parsing, the chunk scheduler (replan trigger, stall, target age), and that importing
the experiments package does not import torch or lerobot. No CUDA, no lerobot, no model weights.

Run (stdlib unittest, no new dependency):
    C:/Users/Admin/.conda/envs/ball_balance_env/python.exe host_software/ml_jetson_vla/tests/test_experiments_cpu.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest

import numpy as np

_THIS_DIR: str = os.path.dirname(os.path.abspath(__file__))
_ML_JETSON_VLA_DIR: str = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_HOST_SOFTWARE_DIR: str = os.path.abspath(os.path.join(_ML_JETSON_VLA_DIR, ".."))
if _HOST_SOFTWARE_DIR not in sys.path:
    sys.path.append(_HOST_SOFTWARE_DIR)

from ml_jetson_vla.core.minimal_vlm_policy import COLOR_COMMANDS  # noqa: E402
from ml_jetson_vla.experiments import orchestrators as orch  # noqa: E402
from ml_jetson_vla.experiments import schedule as sch  # noqa: E402
from ml_jetson_vla.experiments import serial_protocol as sp  # noqa: E402

_STATE: np.ndarray = np.array([10.0, -5.0])
_FRAME: np.ndarray = np.zeros((4, 4, 3), dtype=np.uint8)


def _plan(n_steps: int, inference_ms: float, value: float = 1.0) -> sch.ChunkPlan:
    actions = np.full((n_steps, 3), value, dtype=np.float64)
    return sch.ChunkPlan(
        option="test", weights_status="test", actions_deg=actions,
        safety=sp.validate_chunk_angles(actions), inference_ms=inference_ms,
    )


class StateMachineMapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.o = orch.StateMachineOrchestrator()

    def test_every_colour_command_resolves_to_marker(self) -> None:
        for cmd, label in COLOR_COMMANDS.items():
            req = self.o.resolve(cmd, _FRAME, _STATE)
            self.assertEqual(req.marker_label, label)
            self.assertIsNone(req.fixed_target_mm)

    def test_hold_and_stop_resolve_to_fixed_hold_at_ball(self) -> None:
        for cmd in ("hold", "stop"):
            self.o.reset()
            req = self.o.resolve(cmd, _FRAME, _STATE)
            self.assertIsNone(req.marker_label)
            self.assertEqual(req.fixed_action, "hold")
            self.assertEqual(req.fixed_target_mm, (10.0, -5.0))

    def test_nudges_move_hold_point_with_state_machine_signs(self) -> None:
        self.o.reset()
        base = self.o.resolve("hold", _FRAME, _STATE).fixed_target_mm
        assert base is not None
        fwd = self.o.resolve("forward", _FRAME, _STATE).fixed_target_mm
        assert fwd is not None
        self.assertAlmostEqual(fwd[0], base[0])
        self.assertAlmostEqual(fwd[1], base[1] + orch.NUDGE_FRACTION * orch.PLATFORM_HEIGHT_MM)
        left = self.o.resolve("left", _FRAME, _STATE).fixed_target_mm
        assert left is not None
        self.assertAlmostEqual(left[0], fwd[0] + orch.NUDGE_FRACTION * orch.PLATFORM_WIDTH_MM)

    def test_nudge_is_clamped_to_ninety_percent_half_range(self) -> None:
        self.o.reset()
        far = np.array([0.0, 0.0])
        for _ in range(50):
            req = self.o.resolve("left", _FRAME, far)
        assert req.fixed_target_mm is not None
        self.assertLessEqual(req.fixed_target_mm[0], orch.CLAMP_FRACTION * orch.PLATFORM_WIDTH_MM / 2.0 + 1e-9)

    def test_unknown_instruction_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.o.resolve("go_blue", _FRAME, _STATE)

    def test_target_request_needs_exactly_one_target_kind(self) -> None:
        with self.assertRaises(ValueError):
            orch.TargetRequest("x", None, None, None)
        with self.assertRaises(ValueError):
            orch.TargetRequest("x", "red marker", (0.0, 0.0), None)


class LanguageStubTests(unittest.TestCase):
    def test_llm_stub_raises_not_implemented_with_message(self) -> None:
        lo = orch.LanguageOrchestrator()
        with self.assertRaises(NotImplementedError) as ctx:
            lo.resolve("bring me the green one", _FRAME, _STATE)
        self.assertIn("OPTION L", str(ctx.exception))

    def test_wired_parse_fn_is_used_and_validated(self) -> None:
        lo = orch.LanguageOrchestrator(parse_fn=lambda text: "green marker")
        self.assertEqual(lo.resolve("x", _FRAME, _STATE).marker_label, "green marker")
        bad = orch.LanguageOrchestrator(parse_fn=lambda text: "purple")
        with self.assertRaises(ValueError):
            bad.resolve("x", _FRAME, _STATE)


class SerialProtocolTests(unittest.TestCase):
    def test_format_line(self) -> None:
        self.assertEqual(sp.format_angle_line(1.0, -2.5, 0.0), "A,1.0000,-2.5000,0.0000")

    def test_limit_boundary_accepted_and_just_over_rejected(self) -> None:
        self.assertEqual(sp.format_angle_line(11.025, -11.025, 0.0), "A,11.0250,-11.0250,0.0000")
        with self.assertRaises(sp.AngleOutOfRangeError):
            sp.format_angle_line(11.0251, 0.0, 0.0)
        with self.assertRaises(sp.AngleOutOfRangeError):
            sp.format_angle_line(0.0, -11.0251, 0.0)

    def test_non_finite_rejected(self) -> None:
        for bad in (float("nan"), float("inf"), -float("inf")):
            with self.assertRaises(sp.AngleOutOfRangeError):
                sp.format_angle_line(bad, 0.0, 0.0)

    def test_chunk_boundary_accepted(self) -> None:
        chunk = np.array([[11.025, -11.025, 0.0], [1.0, 2.0, 3.0]])
        safety = sp.validate_chunk_angles(chunk)
        self.assertTrue(safety.accepted)
        self.assertEqual(safety.n_violations, 0)

    def test_chunk_with_one_bad_value_is_wholly_rejected_and_not_clamped(self) -> None:
        chunk = np.array([[1.0, 2.0, 3.0], [1.0, 11.0251, 3.0]])
        safety = sp.validate_chunk_angles(chunk)
        self.assertFalse(safety.accepted)
        self.assertEqual(safety.n_violations, 1)
        self.assertIsNone(sp.release_chunk_lines(chunk))

    def test_empty_chunk_rejected(self) -> None:
        self.assertFalse(sp.validate_chunk_angles(np.zeros((0, 3))).accepted)

    def test_accepted_chunk_lines(self) -> None:
        lines = sp.release_chunk_lines(np.array([[0.0, 0.0, 0.0], [1.5, -1.5, 0.25]]))
        self.assertEqual(lines, ["A,0.0000,0.0000,0.0000", "A,1.5000,-1.5000,0.2500"])

    def test_uplink_parse_converts_hundredths_and_flags_validity(self) -> None:
        s = sp.parse_uplink_line("T,12,3400,-8835,-2940,1,0,-98,33\n")
        assert s is not None
        self.assertAlmostEqual(s.touch_x_mm, -88.35)
        self.assertAlmostEqual(s.touch_y_mm, -29.40)
        self.assertTrue(s.valid)
        self.assertEqual(s.steps, (0, -98, 33))
        self.assertFalse(sp.parse_uplink_line("T,1,2,3,4,0,0,0,0").valid)  # type: ignore[union-attr]

    def test_uplink_skips_comments_and_malformed(self) -> None:
        self.assertIsNone(sp.parse_uplink_line("# hb angle=1"))
        self.assertIsNone(sp.parse_uplink_line("T,1,2,3"))
        self.assertIsNone(sp.parse_uplink_line("T,1,2,x,4,1,0,0,0"))


class ScheduleTests(unittest.TestCase):
    def test_first_replan_triggers_immediately(self) -> None:
        s = sch.ChunkScheduler(control_hz=30.0, margin_ms=20.0)
        self.assertTrue(s.should_replan(0.0))

    def test_replan_triggers_when_remaining_below_inference_plus_margin(self) -> None:
        s = sch.ChunkScheduler(control_hz=30.0, margin_ms=20.0)
        s.begin_inference(0.0)
        s.land(0.05, _plan(10, inference_ms=50.0), None, True)  # chunk 10 steps = 333 ms, from t=0.05
        # estimate 50 ms + margin 20 ms = 70 ms. Remaining at t: end(0.383) - t.
        self.assertFalse(s.should_replan(0.2))   # 183 ms left, not below 70
        self.assertTrue(s.should_replan(0.32))   # 63 ms left, below 70

    def test_no_replan_while_inference_pending(self) -> None:
        s = sch.ChunkScheduler()
        s.begin_inference(0.0)
        self.assertFalse(s.should_replan(10.0))
        with self.assertRaises(RuntimeError):
            s.begin_inference(0.1)

    def test_stall_detected_when_chunk_lands_after_it_ran_out(self) -> None:
        s = sch.ChunkScheduler(control_hz=30.0, margin_ms=20.0)
        s.begin_inference(0.0)
        first = s.land(0.0, _plan(3, 10.0), None, True)  # chunk 3 steps = 100 ms, ends at 0.1
        self.assertTrue(first.stall)  # nothing played before the first chunk
        s.begin_inference(0.05)
        ok = s.land(0.08, _plan(3, 10.0), 40.0, False)  # lands before 0.1: queued, no stall
        self.assertFalse(ok.stall)
        self.assertAlmostEqual(s.chunk_end_s or 0.0, 0.2)  # queued after the first chunk
        s.begin_inference(0.25)
        late = s.land(0.30, _plan(3, 10.0), 55.0, True)  # chunk ended at 0.2, lands at 0.30
        self.assertTrue(late.stall)
        self.assertAlmostEqual(late.stall_gap_ms, 100.0, places=6)

    def test_rejected_chunk_contributes_no_steps_and_causes_stall(self) -> None:
        s = sch.ChunkScheduler()
        s.begin_inference(0.0)
        s.land(0.0, _plan(10, 5.0), None, True)
        s.begin_inference(0.1)
        bad = sch.ChunkPlan("t", "t", np.array([[99.0, 0.0, 0.0]]), sp.validate_chunk_angles(np.array([[99.0, 0.0, 0.0]])), 5.0)
        rec = s.land(0.1, bad, None, False)
        self.assertFalse(rec.accepted)
        # The rejected chunk adds no steps: the earlier 10-step chunk (ends at 10/30 s) is unchanged.
        self.assertAlmostEqual(s.chunk_end_s or 0.0, 10.0 / 30.0)

    def test_target_age(self) -> None:
        self.assertAlmostEqual(sch.target_age_ms(1.0, 1.5), 500.0)
        with self.assertRaises(ValueError):
            sch.target_age_ms(2.0, 1.0)

    def test_summary_counts(self) -> None:
        s = sch.ChunkScheduler()
        s.begin_inference(0.0)
        s.land(0.0, _plan(10, 20.0), None, True)
        s.begin_inference(0.1)
        s.land(0.1, _plan(10, 40.0), 300.0, True)
        summary = sch.summarize_replans(s.records)
        self.assertEqual(summary["n_replans"], 2)
        self.assertEqual(summary["n_new_target_arrivals"], 2)
        self.assertEqual(summary["n_with_target"], 1)
        self.assertAlmostEqual(summary["target_age_ms_p50"], 300.0)


class ImportIsolationTests(unittest.TestCase):
    def test_package_import_does_not_pull_torch_or_lerobot(self) -> None:
        code = (
            "import sys\n"
            "import ml_jetson_vla.experiments\n"
            "import ml_jetson_vla.experiments.orchestrators\n"
            "import ml_jetson_vla.experiments.schedule\n"
            "import ml_jetson_vla.experiments.serial_protocol\n"
            "import ml_jetson_vla.experiments.fast_layers\n"
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
