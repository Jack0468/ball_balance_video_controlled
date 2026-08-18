"""Track 3 (medium class on Jetson, GPU/TensorRT) -- ONNX export step.

`run_eval_expert.py` (the medium-class pipeline: `yolov8_platform_pose_markers_iphone_v1`
+ `mlp_corrector_iphone_v1`) runs both models directly in PyTorch today, not ONNX.
`ml_vision/training/export_to_onnx.py` already owns exporting every checkpoint in this
project to ONNX -- it just didn't cover these two checkpoints yet (item 6, added
alongside this file: YOLO-pose iphone_v1 via its existing `export_model()`, and a new
`export_corrector_mlp_iphone_v1()` for `CorrectorMLP`, since `export_mlp_corrector()`
there is for a differently-shaped class, `MLPCorrectorTime`).

This file does NOT reimplement export logic -- it just calls into `export_to_onnx.py`,
which writes the resulting .onnx files next to their source checkpoints in
`ml_vision/models/...`, exactly where every other exported model in this repo already
lives (e.g. `shared_vision_backbone_v2/shared_vision_backbone_best.onnx`, what Track 1
already loads). A Jetson GPU-provider runtime (not yet built) would point at those same
paths.

Run once (host PC or Jetson, doesn't need GPU) to produce the .onnx files a GPU-provider
Jetson runtime would load -- export correctness doesn't depend on the target device, so
this has been run and verified locally (numeric parity vs. PyTorch, opset check,
onnxruntime CPU load-and-infer), unlike runtime GPU-provider code, which does need a
Jetson to verify.

VERIFIED FINDING (2026-08-18, this environment: torch 2.12.1, onnx 1.22.0): despite both
exporters requesting opset_version=12, only the YOLO-pose export actually landed at
opset 12. `export_corrector_mlp_iphone_v1()`'s CorrectorMLP export landed at opset 18 --
torch's newer dynamo-based ONNX exporter builds at opset 18 and its opset-18->12
downgrade path failed (`onnx.version_converter`: "No Adapter From Version 16 for
Identity"), silently falling back to 18 rather than raising. Numerically the export is
still correct (checked: max abs diff vs. PyTorch ~7.6e-6), so this doesn't block using
the file today -- but opset 18 may exceed what an older JetPack's bundled
onnxruntime/TensorRT actually supports. Check the target JetPack's supported opset range
before trusting this file loads there without modification.
"""

import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ML_VISION_TRAINING_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", "ml_vision", "training"))
if _ML_VISION_TRAINING_DIR not in sys.path:
    sys.path.append(_ML_VISION_TRAINING_DIR)

from export_to_onnx import export_model, export_corrector_mlp_iphone_v1  # noqa: E402


if __name__ == "__main__":
    print("--- Track 3: medium-class ONNX export (iphone_v1) ---\n")
    export_model("../models/yolov8_platform_pose_markers_iphone_v1/weights/best.pt", is_local_path=True)
    export_corrector_mlp_iphone_v1(
        model_path="../models/mlp_corrector_iphone_v1/best_corrector.pth",
    )
    print("Done -- both models exported next to their source checkpoints in ml_vision/models/.")
