"""CPU-only construction checks for `runtime/run_jetson_standalone.py`'s `JetsonExpertPolicy`
(2026-10-14 REOPENED fix wiring).

Unlike `test_small_cnn_grounder_cpu.py`, this constructs a REAL `JetsonExpertPolicy` --
no mocking of `onnxruntime`, `MarkerClassifier`, or the manifest loader. This is possible
(and preferred, matching the verification precedent in docs/PROJECT_LOGBOOK.md's 2026-10-14
"Reverted the manifest-position colour-ID fix..." entry, which reconstructed a real
`JetsonExpertPolicy` the same way) because:
  - the real `shared_vision_backbone_best.onnx` and `hardware/platform_templates/
    ground_truth_manifest.json` files already exist in this checkout (same fact
    `test_small_cnn_grounder_cpu.py`'s own docstring notes for `SmallCnnGroundingPolicy`),
  - `dummy_audio=True` swaps in `KeyboardCommandReceiver` (a daemon thread reading stdin,
    harmless under a test runner -- no real mic/model needed),
  - no camera or serial port is opened by `JetsonExpertPolicy.__init__` itself (that happens
    later, in `main()`, which this test never calls).

Run (stdlib unittest, no new dependency):
    C:/Users/Admin/.conda/envs/ball_balance_env/python.exe host_software/ml_jetson_vla/tests/test_run_jetson_standalone_cpu.py
"""

from __future__ import annotations

import os
import sys
import unittest

_THIS_DIR: str = os.path.dirname(os.path.abspath(__file__))
_ML_JETSON_VLA_DIR: str = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_HOST_SOFTWARE_DIR: str = os.path.abspath(os.path.join(_ML_JETSON_VLA_DIR, ".."))
if _HOST_SOFTWARE_DIR not in sys.path:
    sys.path.append(_HOST_SOFTWARE_DIR)

from ml_jetson_vla.runtime.run_jetson_standalone import JetsonExpertPolicy  # noqa: E402
from host_software.ml_vision.core.color_calibration import AdaptiveColorCalibrator  # noqa: E402


def _build() -> JetsonExpertPolicy:
    """Real construction, dummy_audio=True (no mic/serial/camera needed) -- same minimal-args
    shape `main()` uses, just with the audio/vision hardware-free branches selected."""
    return JetsonExpertPolicy(
        script_dir=_HOST_SOFTWARE_DIR,
        marker_gate_mm=20.0,
        jump_gate_mm=30.0,
        gate_ema_alpha=0.15,
        seed_window=5,
        seed_consistency_mm=15.0,
        lost_frames=30,
        mask_threshold=0.5,
        dummy_audio=True,
    )


class AdaptiveCalibratorWiringTests(unittest.TestCase):
    """2026-10-14 REOPENED fix: JetsonExpertPolicy must construct its own
    AdaptiveColorCalibrator and pass it into MarkerClassifier -- never ground truth,
    never shared/global across instances (see docs/PROJECT_LOGBOOK.md/docs/TODO.md)."""

    def test_constructs_a_real_adaptive_calibrator_wired_into_marker_classifier(self) -> None:
        policy = _build()
        self.assertIsInstance(policy.adaptive_calibrator, AdaptiveColorCalibrator)
        self.assertIs(policy.marker_classifier.adaptive_calibrator, policy.adaptive_calibrator)
        self.assertFalse(policy.adaptive_calibrator.is_calibrated)  # fresh, not warmed up yet

    def test_two_instances_get_two_distinct_calibrators_not_shared_global_state(self) -> None:
        policy1 = _build()
        policy2 = _build()
        self.assertIsNot(policy1.adaptive_calibrator, policy2.adaptive_calibrator)
        self.assertIsNot(policy1.marker_classifier, policy2.marker_classifier)

    def test_reset_does_not_discard_the_calibrator(self) -> None:
        """reset() rebuilds the PredictionGate (per-episode state) -- it must NOT touch the
        adaptive calibrator, which is per-SESSION (one physical sheet mount), not per-episode.
        See the wiring comment in run_jetson_standalone.py for the full rationale."""
        policy = _build()
        calibrator_before = policy.adaptive_calibrator
        policy.reset()
        self.assertIs(policy.adaptive_calibrator, calibrator_before)
        self.assertIs(policy.marker_classifier.adaptive_calibrator, calibrator_before)


if __name__ == "__main__":
    unittest.main()
