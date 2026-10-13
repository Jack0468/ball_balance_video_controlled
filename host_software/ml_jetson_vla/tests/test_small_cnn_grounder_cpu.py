"""CPU-only checks for `core/small_cnn_grounder.py` (2026-10-13).

Mocks the vision pipeline (`onnxruntime.InferenceSession`, `load_manifest_full`, `MarkerClassifier`
at construction time; `estimate_homography_from_aruco`/`warp_to_platform` at `act()`-call time) at
the module's own import site -- same style `test_capture_homography_cpu.py` already uses for
`cv2`/`estimate_homography_from_aruco`/`load_manifest_full`. No real camera, no real ArUco
detection, no real CNN inference.

Construction-time vs. call-time mocking split, deliberately: `SmallCnnGroundingPolicy.__init__`
builds its ONNX session, ArUco lookup, and `MarkerClassifier` ONCE and stores them as instance
attributes, so those three only need to be mocked while `_build()` constructs the policy.
`estimate_homography_from_aruco`/`warp_to_platform` are plain module-level function calls inside
`act()` itself, looked up fresh every call, so those two are mocked per-test around the `act()`
call instead -- re-patching `MarkerClassifier` at that point would have no effect (the instance was
already built).

The real `shared_vision_backbone_best.onnx` file and `ground_truth_manifest.json` happen to exist
in this checkout (used in production by `runtime/run_jetson_standalone.py`), so
`SmallCnnGroundingPolicy()`'s constructor-time `os.path.exists()` check passes without needing to
be mocked -- but every test here still runs with a fully mocked vision pipeline, never a real
inference.

Run (stdlib unittest, no new dependency):
    C:/Users/Admin/.conda/envs/ball_balance_env/python.exe host_software/ml_jetson_vla/tests/test_small_cnn_grounder_cpu.py
"""

from __future__ import annotations

import inspect
import os
import sys
import unittest
from typing import Any, Optional
from unittest import mock

import numpy as np

_THIS_DIR: str = os.path.dirname(os.path.abspath(__file__))
_ML_JETSON_VLA_DIR: str = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_HOST_SOFTWARE_DIR: str = os.path.abspath(os.path.join(_ML_JETSON_VLA_DIR, ".."))
if _HOST_SOFTWARE_DIR not in sys.path:
    sys.path.append(_HOST_SOFTWARE_DIR)

from ml_jetson_vla.core import small_cnn_grounder as scg  # noqa: E402
from ml_jetson_vla.core.policy_interface import Policy, PolicyCommand  # noqa: E402

_MARKERS = [{"id": i, "center_mm": [float(i), float(i)]} for i in range(4)]
_HOMOGRAPHY = np.eye(3)
_FRAME: np.ndarray = np.zeros((480, 640, 3), dtype=np.uint8)
_WARPED: np.ndarray = np.zeros((128, 128, 3), dtype=np.uint8)
_OUT_HW = (128, 128)


class _FakeInput:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeSession:
    """Stand-in for `ort.InferenceSession` -- `.run()` returns fixed CNN outputs regardless of
    input, since every test here controls detections via the mocked `MarkerClassifier` instead."""

    def __init__(self) -> None:
        self._outputs = {
            "ball_xy": np.zeros((1, 2), dtype=np.float32),
            "mask_logits": np.zeros((1, 1, *_OUT_HW), dtype=np.float32),
            "heatmap_logits": np.zeros((1, 1, *_OUT_HW), dtype=np.float32),
        }

    def get_inputs(self) -> list:
        return [_FakeInput("input")]

    def run(self, names: list, _feed: dict) -> list:
        return [self._outputs[n] for n in names]


class _FakeDetection:
    def __init__(self, color: str, x_mm: float, y_mm: float, area_px: float) -> None:
        self.color = color
        self.x_mm = x_mm
        self.y_mm = y_mm
        self.area_px = area_px


class _FakeMarkerClassifier:
    """Stand-in for `MarkerClassifier` -- `classify()` returns a fixed detection list, ignoring
    its real inputs (the mocked CNN outputs above are deliberately meaningless)."""

    def __init__(self, detections: list) -> None:
        self._detections = detections

    def classify(self, _warped_bgr: np.ndarray, _mask_prob: np.ndarray, _heatmap_prob: np.ndarray) -> list:
        return self._detections


def _build(detections: Optional[list] = None) -> Any:
    """Constructs a real `SmallCnnGroundingPolicy`, with its ONNX session/ArUco lookup/
    `MarkerClassifier` all mocked -- these are built once in `__init__` and stored as instance
    attributes, so they only need mocking here, not again per `act()` call."""
    fake_ort = mock.Mock()
    fake_ort.SessionOptions.return_value = mock.Mock()
    fake_ort.InferenceSession.return_value = _FakeSession()
    with mock.patch.object(scg, "ort", fake_ort), \
         mock.patch.object(scg, "load_manifest_full", return_value=(_MARKERS, [], 187.5, 142.0)), \
         mock.patch.object(scg, "MarkerClassifier", side_effect=lambda **kw: _FakeMarkerClassifier(detections or [])):
        return scg.SmallCnnGroundingPolicy()


def _act_patches(homography: Optional[np.ndarray]):
    """Patches only what `act()` looks up fresh on every call: `estimate_homography_from_aruco`
    and `warp_to_platform` (plain module-level function references, re-resolved each call --
    unlike `MarkerClassifier`, which `_build()` above already bound once as an instance
    attribute)."""
    return (
        mock.patch.object(scg, "estimate_homography_from_aruco", return_value=homography),
        mock.patch.object(scg, "warp_to_platform", return_value=(_WARPED, np.eye(3))),
    )


class ConstructionAndProtocolTests(unittest.TestCase):
    def test_constructs_and_satisfies_policy_protocol(self) -> None:
        policy = _build()
        self.assertIn(Policy, type(policy).__mro__)
        self.assertEqual(list(inspect.signature(policy.act).parameters), ["image", "instruction", "state"])
        self.assertEqual(list(inspect.signature(policy.reset).parameters), [])
        policy.reset()
        self.assertEqual(policy.last_debug, {})

    def test_default_marker_manifest_is_aruco_markers_01(self) -> None:
        """2026-10-13 manifest-position colour-ID fix: default must point at the sheet
        confirmed physically mounted that day, and NOT at ground_truth_manifest.json (whose
        own "features" list is empty -- confirmed separately, see small_cnn_grounder.py's
        constructor comment) -- getting this wrong silently reintroduces the cross-sheet
        failure this feature exists to fix."""
        captured: dict = {}

        def _capture_manifest_classifier(**kw: Any) -> Any:
            captured["manifest_path"] = kw.get("manifest_path")
            return _FakeMarkerClassifier([])

        fake_ort = mock.Mock()
        fake_ort.SessionOptions.return_value = mock.Mock()
        fake_ort.InferenceSession.return_value = _FakeSession()
        with mock.patch.object(scg, "ort", fake_ort), \
             mock.patch.object(scg, "load_manifest_full", return_value=(_MARKERS, [], 187.5, 142.0)), \
             mock.patch.object(scg, "MarkerClassifier", side_effect=_capture_manifest_classifier):
            scg.SmallCnnGroundingPolicy()
        self.assertIsNotNone(captured["manifest_path"])
        self.assertTrue(captured["manifest_path"].endswith("aruco_markers_01_manifest.json"))
        self.assertTrue(os.path.exists(captured["manifest_path"]))

    def test_marker_manifest_path_override_is_honoured(self) -> None:
        captured: dict = {}

        def _capture_manifest_classifier(**kw: Any) -> Any:
            captured["manifest_path"] = kw.get("manifest_path")
            return _FakeMarkerClassifier([])

        # A real file must exist at the override path (constructor checks this) -- reuse the
        # real aruco_markers_03 manifest, a real, different, existing file.
        override_path = os.path.join(
            _HOST_SOFTWARE_DIR, "..", "hardware", "platform_templates", "aruco_markers_03_manifest.json",
        )
        fake_ort = mock.Mock()
        fake_ort.SessionOptions.return_value = mock.Mock()
        fake_ort.InferenceSession.return_value = _FakeSession()
        with mock.patch.object(scg, "ort", fake_ort), \
             mock.patch.object(scg, "load_manifest_full", return_value=(_MARKERS, [], 187.5, 142.0)), \
             mock.patch.object(scg, "MarkerClassifier", side_effect=_capture_manifest_classifier):
            scg.SmallCnnGroundingPolicy(marker_manifest_path=override_path)
        self.assertEqual(os.path.abspath(captured["manifest_path"]), os.path.abspath(override_path))

    def test_missing_marker_manifest_raises_file_not_found(self) -> None:
        fake_ort = mock.Mock()
        fake_ort.SessionOptions.return_value = mock.Mock()
        fake_ort.InferenceSession.return_value = _FakeSession()
        with mock.patch.object(scg, "ort", fake_ort), \
             mock.patch.object(scg, "load_manifest_full", return_value=(_MARKERS, [], 187.5, 142.0)):
            with self.assertRaises(FileNotFoundError):
                scg.SmallCnnGroundingPolicy(marker_manifest_path="/does/not/exist_manifest.json")


class UnrecognisedInstructionTests(unittest.TestCase):
    def test_unrecognised_instruction_returns_nan_without_touching_vision(self) -> None:
        policy = _build()
        with mock.patch.object(scg, "estimate_homography_from_aruco") as homography_fn:
            cmd = policy.act(_FRAME, "forward", {})
        homography_fn.assert_not_called()  # directional instruction -- never reaches the vision pipeline
        self.assertTrue(np.isnan(cmd.target_x_mm))
        self.assertTrue(np.isnan(cmd.target_y_mm))
        self.assertFalse(policy.last_debug["parse_ok"])
        self.assertIn("unrecognised_instruction", policy.last_debug["reason"])

    def test_none_instruction_returns_nan(self) -> None:
        policy = _build()
        cmd = policy.act(_FRAME, None, {})
        self.assertTrue(np.isnan(cmd.target_x_mm))
        self.assertFalse(policy.last_debug["parse_ok"])


class HomographyAndDetectionTests(unittest.TestCase):
    def test_no_aruco_detected_degrades_gracefully(self) -> None:
        policy = _build()
        p1, p2 = _act_patches(None)
        with p1, p2:
            cmd = policy.act(_FRAME, "go_red", {})
        self.assertTrue(np.isnan(cmd.target_x_mm))
        self.assertTrue(np.isnan(cmd.target_y_mm))
        self.assertFalse(policy.last_debug["parse_ok"])
        self.assertEqual(policy.last_debug["reason"], "no_aruco")

    def test_no_matching_colour_detected_returns_nan(self) -> None:
        policy = _build(detections=[_FakeDetection("blue", 1.0, 2.0, 50.0)])
        p1, p2 = _act_patches(_HOMOGRAPHY)
        with p1, p2:
            cmd = policy.act(_FRAME, "go_red", {})
        self.assertTrue(np.isnan(cmd.target_x_mm))
        self.assertFalse(policy.last_debug["parse_ok"])
        self.assertEqual(policy.last_debug["reason"], "no_matching_marker_detected")
        self.assertEqual(policy.last_debug["n_detections"], 1)

    def test_matching_marker_returns_its_real_mm_coordinates(self) -> None:
        detections = [
            _FakeDetection("blue", 1.0, 2.0, 999.0),  # larger area, wrong colour -- must not win
            _FakeDetection("red", 42.5, -17.3, 50.0),
        ]
        policy = _build(detections)
        p1, p2 = _act_patches(_HOMOGRAPHY)
        with p1, p2:
            cmd = policy.act(_FRAME, "go_red", {})
        self.assertEqual(cmd.target_x_mm, 42.5)
        self.assertEqual(cmd.target_y_mm, -17.3)
        self.assertTrue(policy.last_debug["parse_ok"])
        self.assertEqual(policy.last_debug["color"], "red")

    def test_largest_area_wins_among_same_colour_detections(self) -> None:
        detections = [
            _FakeDetection("red", 1.0, 1.0, 10.0),
            _FakeDetection("red", 99.0, 99.0, 500.0),  # larger area -- this one should win
            _FakeDetection("red", 2.0, 2.0, 20.0),
        ]
        policy = _build(detections)
        p1, p2 = _act_patches(_HOMOGRAPHY)
        with p1, p2:
            cmd = policy.act(_FRAME, "go_red", {})
        self.assertEqual(cmd.target_x_mm, 99.0)
        self.assertEqual(cmd.target_y_mm, 99.0)

    def test_every_colour_command_maps_to_its_own_colour(self) -> None:
        for instruction, expected_color in (
            ("go_red", "red"), ("go_green", "green"), ("go_yellow", "yellow"), ("go_black", "black"),
        ):
            policy = _build(detections=[_FakeDetection(expected_color, 5.0, 6.0, 30.0)])
            p1, p2 = _act_patches(_HOMOGRAPHY)
            with p1, p2:
                cmd = policy.act(_FRAME, instruction, {})
            self.assertTrue(policy.last_debug["parse_ok"], f"{instruction} should have matched {expected_color}")
            self.assertEqual(cmd.target_x_mm, 5.0)


class PolicyCommandShapeTests(unittest.TestCase):
    def test_act_returns_a_policy_command(self) -> None:
        policy = _build(detections=[_FakeDetection("red", 1.0, 2.0, 10.0)])
        p1, p2 = _act_patches(_HOMOGRAPHY)
        with p1, p2:
            cmd = policy.act(_FRAME, "go_red", {})
        self.assertIsInstance(cmd, PolicyCommand)
        self.assertIsNone(cmd.step_targets)
        self.assertIsNone(cmd.angle_targets_deg)


if __name__ == "__main__":
    unittest.main()
