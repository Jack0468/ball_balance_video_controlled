"""CPU-only checks for the Arm 2 action-head latency benches (2026-10-06).

Covers the pure functions of deployment/bench_action_models.py (latency summary, fits-chunk rule, replan Hz, pixel ->
telemetry mm, the extra-target ACT feature spec, overwrite guard, atomic JSON) and of
deployment/bench_hybrid_qwen_act.py (schedule summary, target store). No CUDA, no lerobot, no model weights.
The one lerobot-dependent test (ACT config construction) skips cleanly when lerobot cannot import.

Run (repo convention: standalone and pytest-discoverable, stdlib unittest, no new dependency):
    C:/Users/Admin/.conda/envs/ball_balance_env/python.exe host_software/ml_jetson_vla/tests/test_bench_action_models_cpu.py
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
import unittest
from typing import Any

import numpy as np

_THIS_DIR: str = os.path.dirname(os.path.abspath(__file__))
_ML_JETSON_VLA_DIR: str = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_HOST_SOFTWARE_DIR: str = os.path.abspath(os.path.join(_ML_JETSON_VLA_DIR, ".."))
if _HOST_SOFTWARE_DIR not in sys.path:
    sys.path.append(_HOST_SOFTWARE_DIR)

from ml_jetson_vla.deployment import bench_action_models as bam  # noqa: E402
from ml_jetson_vla.deployment import bench_hybrid_qwen_act as bhq  # noqa: E402

CONVERTER_PATH: str = os.path.join(_ML_JETSON_VLA_DIR, "data_processing", "convert_to_lerobot.py")


def _affine_homography(scale: float, ox: float, oy: float) -> np.ndarray:
    # mm -> px: px = scale*x_mm + ox, py = scale*y_mm + oy
    return np.array([[scale, 0.0, ox], [0.0, scale, oy], [0.0, 0.0, 1.0]], dtype=np.float64)


class LatencySummaryTests(unittest.TestCase):
    def test_summary_known_values(self) -> None:
        s = bam.summarize_latencies_ms([float(i) for i in range(1, 101)])
        self.assertEqual(s["n"], 100.0)
        self.assertAlmostEqual(s["p50_ms"], 50.5, places=6)
        self.assertAlmostEqual(s["p90_ms"], 90.1, places=6)
        self.assertAlmostEqual(s["p99_ms"], 99.01, places=6)
        self.assertAlmostEqual(s["mean_ms"], 50.5, places=6)

    def test_summary_rejects_empty(self) -> None:
        with self.assertRaises(ValueError):
            bam.summarize_latencies_ms([])

    def test_implied_replan_hz(self) -> None:
        self.assertAlmostEqual(bam.implied_replan_hz(200.0), 5.0, places=9)
        with self.assertRaises(ValueError):
            bam.implied_replan_hz(0.0)


class FitsChunkTests(unittest.TestCase):
    def test_chunk_duration(self) -> None:
        self.assertAlmostEqual(bam.chunk_duration_s(50), 50 / 30.0, places=9)
        self.assertAlmostEqual(bam.chunk_duration_s(100), 100 / 30.0, places=9)

    def test_chunk_duration_rejects_zero(self) -> None:
        with self.assertRaises(ValueError):
            bam.chunk_duration_s(0)

    def test_fits_is_strict_at_boundary(self) -> None:
        boundary_ms = bam.chunk_duration_s(50) * 1000.0
        self.assertFalse(bam.inference_fits_chunk(boundary_ms, 50))
        self.assertTrue(bam.inference_fits_chunk(boundary_ms - 1.0, 50))

    def test_fits_true_and_false_cases(self) -> None:
        self.assertFalse(bam.inference_fits_chunk(2000.0, 50))  # 1.67 s chunk, 2.0 s inference
        self.assertTrue(bam.inference_fits_chunk(3000.0, 100))  # 3.33 s chunk, 3.0 s inference


class PixelToTelemetryTests(unittest.TestCase):
    def test_manifest_to_telemetry_inverts_scorer(self) -> None:
        try:
            scorer = importlib.import_module("ml_jetson_vla.deployment.score_minimal_baseline_offline")
        except ImportError as exc:
            self.skipTest(f"scorer not importable here: {exc}")
        for tx, ty in [(-50.0, 20.0), (10.5, -30.25), (0.0, 0.0)]:
            mx, my = scorer.touch_frame_to_manifest_mm(tx, ty)
            rx, ry = bam.manifest_mm_to_telemetry_mm(mx, my)
            self.assertAlmostEqual(rx, tx, places=9)
            self.assertAlmostEqual(ry, ty, places=9)

    def test_raw_pixel_path_matches_hand_computed_mm(self) -> None:
        try:
            importlib.import_module("ml_jetson_vla.deployment.score_minimal_baseline_offline")
        except ImportError as exc:
            self.skipTest(f"scorer not importable here: {exc}")
        # Manifest point (60, 30) mm -> raw px (340, 170) under scale 4 / offset (100, 50).
        tx, ty = bam.pixel_to_telemetry_mm(_affine_homography(4.0, 100.0, 50.0), 340.0, 170.0,
                                           "raw_image", None, (480, 640))
        ex, ey = bam.manifest_mm_to_telemetry_mm(60.0, 30.0)
        self.assertAlmostEqual(tx, ex, places=3)
        self.assertAlmostEqual(ty, ey, places=3)

    def test_model_input_space_is_rescaled_to_raw(self) -> None:
        try:
            importlib.import_module("ml_jetson_vla.deployment.score_minimal_baseline_offline")
        except ImportError as exc:
            self.skipTest(f"scorer not importable here: {exc}")
        # Model space 504x364 -> raw 640x480: model (252, 182) -> raw (320, 240) -> manifest (55, 47.5) mm.
        tx, ty = bam.pixel_to_telemetry_mm(_affine_homography(4.0, 100.0, 50.0), 252.0, 182.0,
                                           "model_input", (364, 504), (480, 640))
        ex, ey = bam.manifest_mm_to_telemetry_mm(55.0, 47.5)
        self.assertAlmostEqual(tx, ex, places=3)
        self.assertAlmostEqual(ty, ey, places=3)


class ActFeatureSpecTests(unittest.TestCase):
    def test_with_target_extends_state_to_four(self) -> None:
        spec = bam.build_act_feature_spec(True, (480, 640))
        state = spec["input"][bam.STATE_KEY]
        self.assertEqual(tuple(state["shape"]), (4,))
        self.assertEqual(state["names"], ["touch_x_mm", "touch_y_mm", "target_x_mm", "target_y_mm"])
        self.assertEqual(tuple(spec["input"][bam.IMAGE_KEY]["shape"]), (3, 480, 640))
        self.assertEqual(tuple(spec["output"]["action"]["shape"]), (3,))

    def test_without_target_keeps_two_dim_state(self) -> None:
        spec = bam.build_act_feature_spec(False, (480, 640))
        self.assertEqual(tuple(spec["input"][bam.STATE_KEY]["shape"]), (2,))
        self.assertEqual(spec["input"][bam.STATE_KEY]["names"], ["touch_x_mm", "touch_y_mm"])

    def test_state_names_match_converter_schema(self) -> None:
        with open(CONVERTER_PATH, "r", encoding="utf-8") as fh:
            source = fh.read()
        self.assertIn('"names": ["touch_x_mm", "touch_y_mm"]', source)

    def test_act_config_random_backbone_and_state_dim(self) -> None:
        # lerobot 0.4.4's package init fails under transformers 5 (TypeError, not ImportError), so catch broadly here.
        try:
            importlib.import_module("lerobot.policies.act.modeling_act")
        except Exception as exc:
            self.skipTest(f"lerobot.policies not importable in this env: {type(exc).__name__}: {exc}")
        _policy, cfg = bam.build_act_policy(chunk_len=50, with_target=True)
        self.assertIsNone(cfg.pretrained_backbone_weights)
        self.assertEqual(cfg.chunk_size, 50)
        self.assertEqual(cfg.n_action_steps, 50)
        self.assertEqual(tuple(cfg.robot_state_feature.shape), (4,))


class OutputGuardTests(unittest.TestCase):
    def test_refuse_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.json")
            bam.refuse_overwrite(path, force=False)  # absent: fine
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("{}")
            with self.assertRaises(FileExistsError):
                bam.refuse_overwrite(path, force=False)
            bam.refuse_overwrite(path, force=True)

    def test_write_json_atomic_roundtrip(self) -> None:
        payload: dict[str, Any] = {"status": "ok", "values": [1.5, 2.0], "shape": (1, 50, 3)}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "nested", "out.json")
            bam.write_json_atomic(path, payload)
            with open(path, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            self.assertEqual(loaded["status"], "ok")
            self.assertEqual(loaded["values"], [1.5, 2.0])
            self.assertFalse(os.path.exists(path + ".tmp"))


class HybridScheduleTests(unittest.TestCase):
    @staticmethod
    def _rec(has_target: bool, age_ms: Any, fast_ms: float = 100.0) -> dict[str, Any]:
        return {"has_target": has_target, "target_age_ms": age_ms, "new_target_since_last_replan": False,
                "fast_wall_ms": fast_ms, "late_start_ms": 0.0, "output_finite": True}

    def test_stale_fraction_counts_all_replans_and_excludes_no_target(self) -> None:
        records = [
            self._rec(False, None),
            self._rec(True, 500.0),
            self._rec(True, 2500.0),
            self._rec(True, 2000.0),  # exactly 2 s is not stale (strict >)
        ]
        s = bhq.summarize_schedule(records, chunk_len=50)
        self.assertEqual(s["n_replans"], 4)
        self.assertEqual(s["n_no_target"], 1)
        self.assertAlmostEqual(s["fraction_target_older_than_stale_threshold"], 0.25, places=9)
        self.assertAlmostEqual(s["target_age_ms"]["p50_ms"], 2000.0, places=6)

    def test_schedule_summary_rejects_empty(self) -> None:
        with self.assertRaises(ValueError):
            bhq.summarize_schedule([], chunk_len=50)

    def test_target_store_sequence_and_parse_failures(self) -> None:
        store = bhq.TargetStore()
        self.assertIsNone(store.latest())
        store.record_parse_failure(0.5)
        store.put(1.0, 2.0, "s|10", 1.5)
        store.put(3.0, 4.0, "s|11", 1.25)
        latest = store.latest()
        assert latest is not None
        self.assertEqual(latest.seq, 2)
        self.assertEqual((latest.x_mm, latest.y_mm), (3.0, 4.0))
        self.assertEqual(store.slow_calls, 3)
        self.assertEqual(store.parse_failures, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
