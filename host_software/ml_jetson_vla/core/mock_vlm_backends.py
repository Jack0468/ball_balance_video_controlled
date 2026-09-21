"""Mock `VLMBackend`s for dry-running the Arm 2 sweep plumbing without a GPU or any model weights
(`deployment/colab_sweep.py`, `deployment/test_colab_sweep_mock.py`, and the notebook's
`USE_MOCK_BACKENDS` switch).

These are NOT models and never produce results that mean anything about a VLM. Their job is to
exercise the real decode -> homography -> prompt -> parse -> score -> checkpoint/resume path with
outputs in each real candidate's *native output format*, so plumbing bugs surface on a laptop
instead of during a Colab GPU session.

The "oracle": `generate()` locates the requested coloured marker in the image it was handed by
HSV thresholding (independent of the logged ground truth), and reports THAT pixel in the format
under test. Because it uses no telemetry, an error near ~0-7mm through the real scoring path
verifies the corrected ground-truth frame + homography + coordinate-space handling end to end,
while the old (wrong-frame) comparison would score the same outputs 78-170mm off. `go_black` has no
reliable HSV signature here, so the oracle deliberately answers the image centre for it.

Failure/interrupt injection (for fault-isolation and resume tests): `fail_on_load`,
`raise_on_call_indices`, and `interrupt_after_calls` (raises `SimulatedInterrupt`, a
`BaseException` like `KeyboardInterrupt`, so it is NOT swallowed by per-candidate `except
Exception` isolation -- exactly the behaviour a Colab "stop" has).
"""

from __future__ import annotations

import os
import sys
import time
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_HOST_SOFTWARE_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _HOST_SOFTWARE_DIR not in sys.path:
    sys.path.append(_HOST_SOFTWARE_DIR)

from ml_jetson_vla.core.vlm_backends import BackendOutput  # noqa: E402


class SimulatedInterrupt(BaseException):
    """Stands in for KeyboardInterrupt / a Colab runtime stop."""


class MockLoadError(RuntimeError):
    pass


class MockCallError(RuntimeError):
    pass


_COLOR_LABELS = ("red", "green", "yellow")


def _hsv_mask(bgr: np.ndarray, color: str) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    if color == "green":
        return cv2.inRange(hsv, (35, 60, 40), (90, 255, 255))
    if color == "yellow":
        return cv2.inRange(hsv, (18, 60, 120), (34, 255, 255))
    return cv2.inRange(hsv, (0, 90, 80), (8, 255, 255)) | cv2.inRange(hsv, (170, 90, 80), (180, 255, 255))


def locate_marker_px(image_rgb: np.ndarray, target_label: Optional[str]) -> Tuple[float, float]:
    """Centroid (raw-image px) of the requested colour's largest blob inside the central platform
    region of the Track 4 camera view; the image centre if the colour is unknown/not found."""
    h, w = image_rgb.shape[:2]
    centre = (w / 2.0, h / 2.0)
    color = next((c for c in _COLOR_LABELS if target_label and c in target_label), None)
    if color is None:
        return centre
    bgr = cv2.cvtColor(np.ascontiguousarray(image_rgb), cv2.COLOR_RGB2BGR)
    mask = _hsv_mask(bgr, color)
    roi = np.zeros_like(mask)
    x0, x1, y0, y1 = int(0.40 * w), int(0.82 * w), int(0.25 * h), int(0.65 * h)
    roi[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
    n, _lab, stats, cents = cv2.connectedComponentsWithStats(roi)
    if n < 2:
        return centre
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[i, cv2.CC_STAT_AREA] < 40:
        return centre
    return float(cents[i][0]), float(cents[i][1])


class MockBackend:
    """One class, four output flavors (`flavor`): "json" (the asked-for contract, what Qwen /
    InternVL answer), "qwen_point2d" (Qwen's native list), "paligemma_loc" (`<loc>` tokens),
    "moondream_native" (structured `native_points`, normalized then denormalized like the real
    Moondream2Backend). `coord_space="model_input"` + `model_input_hw` makes the JSON/point2d
    flavors answer in a smaller resized space, as the real Qwen backend does."""

    name = "mock"

    def __init__(
        self,
        flavor: str = "json",
        coord_space: str = "raw_image",
        model_input_hw: Optional[Tuple[int, int]] = None,
        dtype: str = "mock",
        fail_on_load: bool = False,
        raise_on_call_indices: Sequence[int] = (),
        interrupt_after_calls: Optional[int] = None,
        sleep_s: float = 0.0,
        mode: str = "n/a",
        **_ignored: object,
    ) -> None:
        if flavor not in ("json", "qwen_point2d", "paligemma_loc", "moondream_native"):
            raise ValueError(f"unknown mock flavor {flavor!r}")
        self.flavor = flavor
        self.coord_space = coord_space
        self.model_input_hw = model_input_hw
        self.dtype_name = dtype
        self.fail_on_load = fail_on_load
        self.raise_on_call_indices = set(raise_on_call_indices)
        self.interrupt_after_calls = interrupt_after_calls
        self.sleep_s = sleep_s
        self.mode = mode
        self._loaded = False
        self.n_calls = 0
        self.load_time_s: Optional[float] = 0.0

    def config_summary(self) -> Dict[str, object]:
        return {"mock": True, "flavor": self.flavor, "coord_space": self.coord_space,
                "model_input_hw": self.model_input_hw, "dtype": self.dtype_name}

    def load(self) -> None:
        if self._loaded:
            return
        if self.fail_on_load:
            raise MockLoadError("simulated load failure (e.g. gated repo / bad remote code)")
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def generate(self, image: np.ndarray, prompt: str, target_label: Optional[str] = None) -> BackendOutput:
        self.load()
        if self.sleep_s:
            time.sleep(self.sleep_s)
        idx = self.n_calls
        self.n_calls += 1
        if self.interrupt_after_calls is not None and idx >= self.interrupt_after_calls:
            raise SimulatedInterrupt(f"simulated interrupt on call {idx}")
        if idx in self.raise_on_call_indices:
            raise MockCallError(f"simulated per-call failure on call {idx}")

        h, w = image.shape[:2]
        x, y = locate_marker_px(image, target_label)
        native_points = None
        if self.flavor == "moondream_native":
            raw = repr({"points": [{"x": x / w, "y": y / h}]})
            native_points = [(x, y)]
        elif self.flavor == "paligemma_loc":
            def tok(v: float, span: float) -> str:
                return f"<loc{min(1023, max(0, int(round(v / span * 1024)))):04d}>"
            raw = (tok(y - 10, h) + tok(x - 10, w) + tok(y + 10, h) + tok(x + 10, w)
                   + f" {target_label or 'target'}")
        else:
            ox, oy = x, y
            if self.coord_space == "model_input" and self.model_input_hw:
                ox, oy = x * self.model_input_hw[1] / w, y * self.model_input_hw[0] / h
            if self.flavor == "json":
                raw = f'{{"target_point_xy": [{int(round(ox))}, {int(round(oy))}]}}'
            else:
                raw = f'[{{"point_2d": [{int(round(ox))}, {int(round(oy))}], "label": "{target_label}"}}]'
        return BackendOutput(
            raw_text=raw, latency_s=0.001, load_time_s=0.0, peak_memory_gb=None,
            backend_name=f"mock:{self.flavor}", native_points=native_points, dtype=self.dtype_name,
            coord_space=self.coord_space, model_input_hw=self.model_input_hw,
        )


MOCK_REGISTRY = {"mock": MockBackend}
