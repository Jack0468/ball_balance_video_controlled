"""Non-Qwen grounder: resolves a colour-marker command to a real (x_mm, y_mm) target using
Track 1's already-validated, already-production small-model pipeline (ArUco homography ->
`shared_vision_backbone_v2` CNN -> `MarkerClassifier`), instead of prompting a VLM.

**Why this exists (debugging/staging tool, NOT a new comparison arm):** `core/hybrid_policy.py`'s
`HybridQwenActPolicy` grounds colour commands with Qwen2.5-VL-3B via `core/minimal_vlm_policy.py`'s
`MinimalVLMPolicy` -- real, but noisy (~60-93% hit@20mm depending on task, measured this session,
see `docs/PROJECT_LOGBOOK.md`) and slow (~1.7-2s/call, GPU-shared with ACT). ACT
(`experiments/fast_layers.py`'s `ActFastLayer`) is itself not trained yet. Testing/training ACT's
own control behaviour against Qwen's noisy/slow grounding confounds two unvalidated things at
once. This module swaps the grounding half for the small-class Shared Backbone CNN +
`MarkerClassifier` pipeline that `runtime/run_jetson_standalone.py`'s `JetsonExpertPolicy` already
runs in production (sub-10ms, already validated) -- so ACT can be exercised against fast, reliable
targets, isolating ACT's own behaviour from Qwen's. **This is explicitly a debugging/staging
configuration for validating ACT in isolation -- it must never be confused with the real Arm 1
(small-model expert pipeline, standalone) vs Arm 2 (Qwen-driven) comparison** (see
`docs/ARCHITECTURE.md`'s 3-arm framing); it is a fourth, informal combination built only to give
ACT a cheap, trustworthy grounding signal to train/test against.

**Integration, mirrored exactly from `runtime/run_jetson_standalone.py`'s `JetsonExpertPolicy.act()`
(the authoritative, already-running-in-production reference for how these pieces fit together --
not re-derived):** load the ArUco manifest once (`load_manifest_full`), build the mm->px
homography per frame (`estimate_homography_from_aruco`), warp to the platform's top-down view
(`warp_to_platform`), run the shared CNN (`preprocess_warped` -> `ball_xy`/`mask_logits`/
`heatmap_logits`), classify blobs into coloured markers (`MarkerClassifier.classify()`, which
already returns `(x_mm, y_mm)` in the firmware/PID touch-mm convention -- no separate
`pixel_to_mm` callable is needed here, unlike `MinimalVLMPolicy`, since `MarkerClassifier` already
does that conversion internally). Among detections of the requested colour, the largest-area one
is used -- the same "largest blob wins per colour" tie-break `JetsonExpertPolicy.act()` already
applies when building `marker_coords`.

**Satisfies `core/policy_interface.py`'s `Policy` protocol**, the same shape `MinimalVLMPolicy`
and `JetsonExpertPolicy` already implement, specifically so `core/hybrid_policy.py`'s `_Grounder`
(generalised to accept any `Policy`-shaped grounding implementation, not just `MinimalVLMPolicy`
-- see that module's docstring) can wrap this one unchanged, reusing its existing
threading/queue/`TargetStore` machinery rather than re-deriving it.

**Output contract, matching `MinimalVLMPolicy`'s so `_Grounder`'s existing ok-check
(`debug.get("parse_ok")` + `np.isfinite(target_x_mm/y_mm)`) works unchanged, with no code path
there needing to know which grounder produced the result:**
  - `instruction` must be one of `core.minimal_vlm_policy.COLOR_COMMANDS`'s keys (e.g. `"go_red"`)
    -- the vocabulary is reused from there, not redefined, so it cannot drift. This grounder only
    ever needs to resolve colour commands: `experiments/orchestrators.py`'s
    `StateMachineOrchestrator` never routes a directional/hold/stop instruction to a grounder at
    all (those resolve to `fixed_target_mm` with no grounding call). Anything else passed to
    `act()` directly is reported as `parse_ok=False` with a reason, never guessed.
  - `debug["parse_ok"]` is `True` only when (a) at least 4 ArUco markers were detected (homography
    succeeded) and (b) `MarkerClassifier` found at least one blob of the requested colour this
    frame. `False` (NaN target, never a stale or fabricated point) otherwise -- "no detection yet"
    (e.g. before the first good frame, or the marker is occluded) is reported honestly, matching
    this project's "no fabricated value on failure" convention used throughout
    `core/minimal_vlm_policy.py` and `JetsonExpertPolicy`.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ML_JETSON_VLA_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_HOST_SOFTWARE_DIR = os.path.abspath(os.path.join(_ML_JETSON_VLA_DIR, ".."))
_REPO_ROOT_DIR = os.path.abspath(os.path.join(_HOST_SOFTWARE_DIR, ".."))
for _p in (_HOST_SOFTWARE_DIR, _REPO_ROOT_DIR):
    if _p not in sys.path:
        sys.path.append(_p)

from ml_jetson_vla.core.minimal_vlm_policy import COLOR_COMMANDS  # noqa: E402
from ml_jetson_vla.core.policy_interface import Policy, PolicyCommand  # noqa: E402

from host_software.ml_vision.core.marker_classifier import MarkerClassifier  # noqa: E402
from host_software.ml_vision.data_processing.auto_label_shared_vision import (  # noqa: E402
    build_paper_corners,
    estimate_homography_from_aruco,
    load_manifest_full,
    warp_to_platform,
)

# NOT imported from main_onnx_shared_vision_audio.py, despite that being the authoritative
# production source for these 5 names: that module's own top level also does
# `from src.audio_receiver_onnx import AudioCommandReceiverONNX`, which needs `sounddevice` --
# a system audio library with no relevance to vision grounding and not installed in the
# arm2-lerobot Docker image (confirmed by a real `ModuleNotFoundError` on the Jetson, 2026-10-13).
# Python import semantics mean `from X import a, b` still runs ALL of X's top-level code, so
# there is no way to pull in just these 5 names without the audio import chain. Copied verbatim
# instead (not re-derived) -- same values/logic as main_onnx_shared_vision_audio.py, checked
# directly against that file, not assumed.
INPUT_SIZE = (128, 128)  # (H, W), must match training
GROUND_TRUTH_MANIFEST = os.path.join(_REPO_ROOT_DIR, "hardware", "platform_templates", "ground_truth_manifest.json")
PAPER_MARGIN_MM = 6.0


def preprocess_warped(warped_bgr: np.ndarray) -> np.ndarray:
    """Matches build_eval_transform() + SharedVisionDataset.__getitem__ exactly:
    RGB, [0,1] float, CHW, batch dim. No ImageNet mean/std normalization --
    that was the OLD model's convention, not this one's."""
    rgb = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2RGB)
    img_f = rgb.astype(np.float32) / 255.0
    chw = np.transpose(img_f, (2, 0, 1))
    return np.expand_dims(chw, axis=0)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

_DEFAULT_CNN_RELPATH = os.path.join(
    "ml_vision", "models", "shared_vision_backbone_v2", "shared_vision_backbone_best.onnx"
)


class SmallCnnGroundingPolicy(Policy):
    """Grounds a colour command via the small-class CNN + ArUco + `MarkerClassifier` pipeline.
    See module docstring for the full contract. Construction loads the ONNX session + ArUco
    manifest once; `act()` is one frame, matching `JetsonExpertPolicy.act()`'s per-call shape."""

    def __init__(
        self,
        host_software_dir: Optional[str] = None,
        cnn_path: Optional[str] = None,
        mask_threshold: float = 0.5,
    ) -> None:
        host_software_dir = host_software_dir or _HOST_SOFTWARE_DIR
        resolved_cnn_path = os.path.abspath(cnn_path or os.path.join(host_software_dir, _DEFAULT_CNN_RELPATH))
        if not os.path.exists(resolved_cnn_path):
            raise FileNotFoundError(f"ONNX vision model not found at {resolved_cnn_path}")

        cnn_opts = ort.SessionOptions()
        cnn_opts.intra_op_num_threads = 2
        cnn_opts.inter_op_num_threads = 1
        self.cnn_session = ort.InferenceSession(resolved_cnn_path, sess_options=cnn_opts, providers=["CPUExecutionProvider"])
        self.cnn_input_name = self.cnn_session.get_inputs()[0].name
        self.marker_classifier = MarkerClassifier(input_size=INPUT_SIZE, mask_threshold=mask_threshold)

        aruco_markers, _features, platform_w_mm, platform_h_mm = load_manifest_full(GROUND_TRUTH_MANIFEST)
        self.aruco_lookup = {int(m["id"]): list(m["center_mm"]) for m in aruco_markers}
        self.paper_corners = build_paper_corners(platform_w_mm, platform_h_mm, margin_mm=PAPER_MARGIN_MM)
        self.platform_w_mm = platform_w_mm
        self.platform_h_mm = platform_h_mm
        self.last_debug: dict = {}

    def reset(self) -> None:
        # No cross-call state here (unlike JetsonExpertPolicy's PredictionGate -- this grounder
        # has no EMA/seed-window to reset). Present to satisfy the Policy protocol.
        self.last_debug = {}

    def act(self, image: np.ndarray, instruction: Optional[str], state: dict) -> PolicyCommand:
        if instruction not in COLOR_COMMANDS:
            debug = {
                "parse_ok": False,
                "reason": f"unrecognised_instruction {instruction!r}; this grounder only resolves "
                          f"colour commands ({sorted(COLOR_COMMANDS)})",
                "instruction": instruction,
            }
            self.last_debug = debug
            return PolicyCommand(target_x_mm=float("nan"), target_y_mm=float("nan"), debug=debug)
        color = COLOR_COMMANDS[instruction].split(" ")[0]  # "red marker" -> "red"

        aruco_homography = estimate_homography_from_aruco(image, self.aruco_lookup)
        if aruco_homography is None:
            debug = {"parse_ok": False, "reason": "no_aruco", "instruction": instruction, "color": color}
            self.last_debug = debug
            return PolicyCommand(target_x_mm=float("nan"), target_y_mm=float("nan"), debug=debug)

        warped_bgr, _warp_matrix = warp_to_platform(
            image, aruco_homography, output_size=INPUT_SIZE[::-1], paper_corners=self.paper_corners
        )
        input_tensor = preprocess_warped(warped_bgr)
        _ball_xy, mask_logits, heatmap_logits = self.cnn_session.run(
            ["ball_xy", "mask_logits", "heatmap_logits"], {self.cnn_input_name: input_tensor}
        )
        mask_prob = sigmoid(mask_logits[0, 0])
        heatmap_prob = sigmoid(heatmap_logits[0, 0])
        detections = self.marker_classifier.classify(warped_bgr, mask_prob, heatmap_prob)

        # Largest-area blob wins per colour -- same tie-break JetsonExpertPolicy.act() already
        # applies when building its own marker_coords dict.
        best = None
        for det in detections:
            if det.color == color and (best is None or det.area_px > best.area_px):
                best = det

        debug = {
            "instruction": instruction, "color": color, "n_detections": len(detections),
            "detected_colors": [det.color for det in detections],
        }
        if best is None:
            debug["parse_ok"] = False
            debug["reason"] = "no_matching_marker_detected"
            self.last_debug = debug
            return PolicyCommand(target_x_mm=float("nan"), target_y_mm=float("nan"), debug=debug)

        debug["parse_ok"] = True
        debug["detection_area_px"] = best.area_px
        self.last_debug = debug
        return PolicyCommand(target_x_mm=best.x_mm, target_y_mm=best.y_mm, debug=debug)
