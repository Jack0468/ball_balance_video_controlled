"""Compares a YOLOv8-pose OpenVINO export against its original .pt-derived export.

Historically a bare top-level script (no test_* function) that constructed two
`ultralytics.YOLO` objects at import time using paths relative to the process's
current working directory -- pytest tried to collect it as a module and
crashed immediately, aborting the ENTIRE `ml_vision/tests/` run with a
collection error rather than a normal failed/skipped test (see
docs/PROJECT_LOGBOOK.md and docs/TODO.md's "pytest collection-abort" entry).

Neither `ml_vision/models/yolov8_platform_markers_v1/` nor `_v2/` has ever
existed in this repo (checked directly: not in the current `models/` listing,
not in `models/archive/`, no commit in `git log --all` ever added either) --
retired-but-kept-runnable rather than deleted, since comparing a .pt export
against its OpenVINO conversion is still a reasonable regression check once
real exports exist again. Wrapped in a real test function with a
`skipif`-on-missing-artifact guard and paths resolved relative to the repo
root (not the cwd) so it is no longer fragile to where pytest is invoked from.
"""

import os

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
MODELS_DIR = os.path.join(REPO_ROOT, "host_software", "ml_vision", "models")
YOLO_OV_DIR = os.path.join(MODELS_DIR, "yolov8_platform_markers_v2", "weights", "best_openvino_model")
YOLO_PT_DIR = os.path.join(MODELS_DIR, "yolov8_platform_markers_v1", "weights", "best_openvino_model")
SAMPLE_IMAGE = os.path.join(REPO_ROOT, "host_software", "data", "yolo_raw_dataset", "images", "0000.jpg")


@pytest.mark.skipif(
    not (os.path.isdir(YOLO_OV_DIR) and os.path.isdir(YOLO_PT_DIR) and os.path.exists(SAMPLE_IMAGE)),
    reason=f"OpenVINO export pair / sample image not present ({YOLO_OV_DIR}, {YOLO_PT_DIR}, {SAMPLE_IMAGE}) -- see module docstring",
)
def test_yolo_openvino_export_matches_pt_export():
    import cv2
    from ultralytics import YOLO

    yolo_ov = YOLO(YOLO_OV_DIR, task="pose")
    yolo_pt = YOLO(YOLO_PT_DIR, task="pose")

    img = cv2.imread(SAMPLE_IMAGE)
    assert img is not None, f"Could not read sample image at {SAMPLE_IMAGE}"

    res_ov = yolo_ov.predict(source=img, imgsz=640, conf=0.5, verbose=False)[0]
    res_pt = yolo_pt.predict(source=img, imgsz=640, conf=0.5, verbose=False)[0]

    cls_ov = res_ov.boxes.cls.cpu().numpy() if res_ov.boxes is not None else None
    cls_pt = res_pt.boxes.cls.cpu().numpy() if res_pt.boxes is not None else None

    # The real point of this comparison: the OpenVINO export should detect the
    # same classes as the original .pt-derived export on the same frame, not
    # merely "both ran without crashing".
    assert cls_ov is not None and cls_pt is not None
    assert sorted(cls_ov.tolist()) == sorted(cls_pt.tolist())
