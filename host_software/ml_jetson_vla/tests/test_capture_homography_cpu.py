"""CPU-only checks for experiments/capture_homography.py (2026-10-09).

Mocks cv2.VideoCapture and estimate_homography_from_aruco/load_manifest_full at the module's
own import site -- no real camera or ArUco detection needed. No CUDA, no lerobot, no model weights.

Run (stdlib unittest, no new dependency):
    C:/Users/Admin/.conda/envs/ball_balance_env/python.exe host_software/ml_jetson_vla/tests/test_capture_homography_cpu.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

_THIS_DIR: str = os.path.dirname(os.path.abspath(__file__))
_ML_JETSON_VLA_DIR: str = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_HOST_SOFTWARE_DIR: str = os.path.abspath(os.path.join(_ML_JETSON_VLA_DIR, ".."))
if _HOST_SOFTWARE_DIR not in sys.path:
    sys.path.append(_HOST_SOFTWARE_DIR)

from ml_jetson_vla.experiments import capture_homography as ch  # noqa: E402

_HOMOGRAPHY: np.ndarray = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 20.0], [0.0, 0.0, 1.0]])
_MARKERS = [{"id": i, "center_mm": [float(i), float(i)]} for i in range(4)]
_FRAME: np.ndarray = np.zeros((4, 4, 3), dtype=np.uint8)


class _FakeCapture:
    """Stand-in for cv2.VideoCapture: `reads` is a list of (ok, frame_or_None) per .read() call."""

    def __init__(self, reads: list, opened: bool = True) -> None:
        self._reads = list(reads)
        self._opened = opened
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802 (matches cv2's own method name)
        return self._opened

    def read(self):
        if not self._reads:
            return False, None
        return self._reads.pop(0)

    def release(self) -> None:
        self.released = True


def _patched(reads: list, opened: bool = True, markers_per_call=None):
    """Context manager patching cv2.VideoCapture, load_manifest_full and
    estimate_homography_from_aruco at capture_homography's own module namespace.
    `markers_per_call`: list of return values for successive estimate_homography_from_aruco calls
    (one per real frame read); defaults to always returning _HOMOGRAPHY."""
    cap = _FakeCapture(reads, opened=opened)
    calls = {"n": 0}

    def _estimate(frame, lookup):
        if markers_per_call is None:
            return _HOMOGRAPHY
        val = markers_per_call[calls["n"]]
        calls["n"] += 1
        return val

    return (
        mock.patch.object(ch, "cv2", mock.Mock(VideoCapture=mock.Mock(return_value=cap))),
        mock.patch.object(ch, "load_manifest_full", return_value=(_MARKERS, [], 187.5, 142.0)),
        mock.patch.object(ch, "estimate_homography_from_aruco", side_effect=_estimate),
        cap,
    )


class CaptureHomographyTests(unittest.TestCase):
    def test_success_first_frame(self) -> None:
        p1, p2, p3, cap = _patched(reads=[(True, _FRAME)])
        with p1, p2, p3:
            result = ch.capture_homography(camera_index=0, n_attempts=10)
        np.testing.assert_array_equal(result, _HOMOGRAPHY)
        self.assertTrue(cap.released)

    def test_retries_on_failed_read(self) -> None:
        p1, p2, p3, cap = _patched(reads=[(False, None), (False, None), (True, _FRAME)])
        with p1, p2, p3:
            result = ch.capture_homography(camera_index=0, n_attempts=10)
        np.testing.assert_array_equal(result, _HOMOGRAPHY)
        self.assertTrue(cap.released)

    def test_retries_on_no_markers_detected(self) -> None:
        reads = [(True, _FRAME)] * 3
        p1, p2, p3, cap = _patched(reads=reads, markers_per_call=[None, None, _HOMOGRAPHY])
        with p1, p2, p3:
            result = ch.capture_homography(camera_index=0, n_attempts=10)
        np.testing.assert_array_equal(result, _HOMOGRAPHY)
        self.assertTrue(cap.released)

    def test_raises_after_exhausting_attempts(self) -> None:
        reads = [(True, _FRAME)] * 3
        p1, p2, p3, cap = _patched(reads=reads, markers_per_call=[None, None, None])
        with p1, p2, p3:
            with self.assertRaisesRegex(RuntimeError, "could not detect"):
                ch.capture_homography(camera_index=0, n_attempts=3)
        # The real camera handle must still be released even on the failure path.
        self.assertTrue(cap.released)

    def test_camera_open_failure(self) -> None:
        p1, p2, p3, _cap = _patched(reads=[], opened=False)
        with p1, p2, p3:
            with self.assertRaisesRegex(RuntimeError, "cannot open camera"):
                ch.capture_homography(camera_index=0, n_attempts=10)

    def test_main_writes_npy(self) -> None:
        p1, p2, p3, _cap = _patched(reads=[(True, _FRAME)])
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "nested", "live_homography.npy")
            with p1, p2, p3:
                rc = ch.main(["--camera-index", "0", "--out", out_path])
            self.assertEqual(rc, 0)
            self.assertTrue(os.path.exists(out_path))
            np.testing.assert_array_equal(np.load(out_path), _HOMOGRAPHY)


if __name__ == "__main__":
    unittest.main()
