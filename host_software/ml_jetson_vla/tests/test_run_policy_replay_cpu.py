"""CPU-only checks for `experiments/run_policy_replay.py` (2026-10-09).

Covers the real capability gap this file closed: `build_policy()` used to be hardcoded to
`StubFastLayer` with no way to override it, so `HybridQwenActPolicy`/`SmolVLADirectPolicy` were
never replayed against anything else. These tests confirm:

  - `--fast-layer` defaults to `"stub"` (omitted entirely, `parse_args()` still produces
    `fast_layer == "stub"`) and `build_policy()`'s `stub` branch constructs a `StubFastLayer`-backed
    policy identical to the pre-existing behaviour (no regression for every existing caller/script
    that does not pass `--fast-layer`).
  - `--fast-layer act`/`--fast-layer smolvla` parse correctly and `build_policy()` reaches the
    `ActFastLayer`/`SmolVLAFastLayer` branch with the right constructor arguments, WITHOUT actually
    importing torch/lerobot -- `ActFastLayer`/`SmolVLAFastLayer` are monkeypatched at the
    `run_policy_replay` module's import site, the same "duck-typed stand-in, no real background
    thread/model" style `test_hybrid_policy_cpu.py` already uses for `_Grounder`/`TargetStore`, so
    this runs on the plain Windows interpreter (no GPU, no Docker image).
  - `--chunk-len` is NOT forwarded to the `smolvla` branch (`SmolVLAFastLayer` derives its own
    chunk length from the pinned checkpoint's config -- same mismatch `run_experiment.py`'s
    `build_fast_layer()` already handles by omitting it for this option).

Run (stdlib unittest, no new dependency):
    C:/Users/Admin/.conda/envs/ball_balance_env/python.exe host_software/ml_jetson_vla/tests/test_run_policy_replay_cpu.py
"""

from __future__ import annotations

import os
import sys
import unittest
from typing import Any, Optional
from unittest import mock

_THIS_DIR: str = os.path.dirname(os.path.abspath(__file__))
_ML_JETSON_VLA_DIR: str = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_HOST_SOFTWARE_DIR: str = os.path.abspath(os.path.join(_ML_JETSON_VLA_DIR, ".."))
if _HOST_SOFTWARE_DIR not in sys.path:
    sys.path.append(_HOST_SOFTWARE_DIR)

from ml_jetson_vla.core.hybrid_policy import HybridQwenActPolicy, SmolVLADirectPolicy  # noqa: E402
from ml_jetson_vla.experiments import run_policy_replay as rpr  # noqa: E402
from ml_jetson_vla.experiments.fast_layers import StubFastLayer  # noqa: E402


class _FakeActFastLayer:
    """Duck-typed stand-in for `ActFastLayer` -- records the constructor args it was called with
    and exposes the same `name`/`weights_status`/`plan()` shape, with no torch import."""

    name: str = "act_fast"

    def __init__(self, chunk_len: int, device: str = "cpu", checkpoint_dir: Optional[str] = None) -> None:
        self.chunk_len = chunk_len
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.weights_status = (
            "RANDOM_INIT: no trained ACT checkpoint; actions are not meaningful"
            if checkpoint_dir is None
            else f"TRAINED checkpoint {checkpoint_dir}; normaliser stats not verified"
        )

    def plan(self, target_mm: Any, state_mm: Any, frame: Any, instruction: Optional[str] = None) -> Any:
        raise AssertionError("plan() should not be called by these construction-only tests")


class _FakeSmolVLAFastLayer:
    """Duck-typed stand-in for `SmolVLAFastLayer` -- `chunk_len` is deliberately NOT a constructor
    parameter here, matching the real class (it derives its own from the pinned checkpoint)."""

    name: str = "smolvla_direct"

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.weights_status = "PRETRAINED lerobot/smolvla_base@pinned; NOT fine-tuned"


class BuildPolicyDefaultIsStubTests(unittest.TestCase):
    """`--fast-layer` omitted / defaulted must produce exactly the old StubFastLayer behaviour."""

    def test_parse_args_defaults_fast_layer_to_stub(self) -> None:
        args = rpr.parse_args(
            ["--policy", "hybrid_qwen_act", "--session", "unused", "--out", "unused.json"]
        )
        self.assertEqual(args.fast_layer, "stub")
        self.assertEqual(args.device, "cpu")
        self.assertIsNone(args.act_checkpoint)

    def test_build_policy_default_backs_hybrid_with_stub(self) -> None:
        policy = rpr.build_policy("hybrid_qwen_act", chunk_len=5, margin_ms=40.0)
        try:
            self.assertIsInstance(policy, HybridQwenActPolicy)
            self.assertIn("STUB", policy.weights_status)
            self.assertIsInstance(policy._fast.fast_layer, StubFastLayer)
        finally:
            policy.close()

    def test_build_policy_explicit_stub_backs_smolvla_direct_identically(self) -> None:
        policy = rpr.build_policy(
            "smolvla_direct", chunk_len=5, margin_ms=40.0, fast_layer="stub", device="cpu", act_checkpoint=None
        )
        try:
            self.assertIsInstance(policy, SmolVLADirectPolicy)
            self.assertIsInstance(policy._fast.fast_layer, StubFastLayer)
            self.assertEqual(policy._fast.fast_layer._chunk_len, 5)
        finally:
            policy.close()


class BuildPolicyActAndSmolVLABranchesTests(unittest.TestCase):
    """`--fast-layer act`/`smolvla` must reach the right branch with the right constructor args,
    verified via monkeypatched stand-ins so this runs without torch/lerobot on this interpreter."""

    def test_fast_layer_act_choice_parses(self) -> None:
        args = rpr.parse_args(
            [
                "--policy", "hybrid_qwen_act", "--session", "unused", "--out", "unused.json",
                "--fast-layer", "act", "--device", "cuda", "--act-checkpoint", "/ckpt/dir",
            ]
        )
        self.assertEqual(args.fast_layer, "act")
        self.assertEqual(args.device, "cuda")
        self.assertEqual(args.act_checkpoint, "/ckpt/dir")

    def test_fast_layer_smolvla_choice_parses(self) -> None:
        args = rpr.parse_args(
            [
                "--policy", "smolvla_direct", "--session", "unused", "--out", "unused.json",
                "--fast-layer", "smolvla", "--device", "cuda",
            ]
        )
        self.assertEqual(args.fast_layer, "smolvla")
        self.assertEqual(args.device, "cuda")

    def test_invalid_fast_layer_choice_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            rpr.parse_args(
                [
                    "--policy", "hybrid_qwen_act", "--session", "unused", "--out", "unused.json",
                    "--fast-layer", "bogus",
                ]
            )

    def test_build_policy_act_branch_constructs_with_chunk_len_device_checkpoint(self) -> None:
        with mock.patch.object(rpr, "ActFastLayer", _FakeActFastLayer):
            policy = rpr.build_policy(
                "hybrid_qwen_act", chunk_len=7, margin_ms=40.0,
                fast_layer="act", device="cuda", act_checkpoint="/some/ckpt",
            )
        try:
            self.assertIsInstance(policy, HybridQwenActPolicy)
            layer = policy._fast.fast_layer
            self.assertIsInstance(layer, _FakeActFastLayer)
            self.assertEqual(layer.chunk_len, 7)
            self.assertEqual(layer.device, "cuda")
            self.assertEqual(layer.checkpoint_dir, "/some/ckpt")
            self.assertIn("TRAINED checkpoint", policy.weights_status)
            self.assertTrue(policy.trained_for_task)
        finally:
            policy.close()

    def test_build_policy_act_branch_random_init_when_no_checkpoint(self) -> None:
        with mock.patch.object(rpr, "ActFastLayer", _FakeActFastLayer):
            policy = rpr.build_policy(
                "hybrid_qwen_act", chunk_len=7, margin_ms=40.0, fast_layer="act", device="cpu", act_checkpoint=None
            )
        try:
            self.assertIn("RANDOM_INIT", policy.weights_status)
            self.assertFalse(policy.trained_for_task)
        finally:
            policy.close()

    def test_build_policy_smolvla_branch_does_not_forward_chunk_len(self) -> None:
        """SmolVLAFastLayer has no `chunk_len` parameter -- passing it would be a TypeError, so
        this also guards against a future regression that tries to pass it through."""
        with mock.patch.object(rpr, "SmolVLAFastLayer", _FakeSmolVLAFastLayer):
            policy = rpr.build_policy(
                "smolvla_direct", chunk_len=99, margin_ms=40.0, fast_layer="smolvla", device="cuda"
            )
        try:
            self.assertIsInstance(policy, SmolVLADirectPolicy)
            layer = policy._fast.fast_layer
            self.assertIsInstance(layer, _FakeSmolVLAFastLayer)
            self.assertEqual(layer.device, "cuda")
            self.assertFalse(hasattr(layer, "chunk_len"))
            self.assertIn("PRETRAINED", policy.weights_status)
        finally:
            policy.close()


if __name__ == "__main__":
    unittest.main()
