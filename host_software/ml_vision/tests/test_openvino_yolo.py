"""OpenVINO smoke-check for a YOLOv8 marker-pose export.

Historically a bare top-level script (no test_* function), so pytest tried to
collect it as a module and crashed at import time the moment the referenced
export didn't exist on disk -- aborting the ENTIRE `ml_vision/tests/` run with
a collection error rather than a normal failed/skipped test (see
docs/PROJECT_LOGBOOK.md and docs/TODO.md's "pytest collection-abort" entry).

`ml_vision/models/yolov8_platform_markers_v2/weights/best_openvino_model/` has
never existed in this repo (checked directly: not in the current `models/`
listing, not in `models/archive/`, and no commit in `git log --all` ever added
it) -- this test is retired-but-kept-runnable rather than deleted, since the
OpenVINO-export sanity check itself is still a reasonable thing to want once a
real export exists again. Wrapped in a real test function with a
`skipif`-on-missing-artifact guard: skips cleanly today, and will actually run
end-to-end the moment that export path is restored.
"""

import os

import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_XML = os.path.abspath(
    os.path.join(
        SCRIPT_DIR,
        "ml_vision",
        "models",
        "yolov8_platform_markers_v2",
        "weights",
        "best_openvino_model",
        "best.xml",
    )
)


@pytest.mark.skipif(
    not os.path.exists(YOLO_XML),
    reason=f"OpenVINO export not present at {YOLO_XML} -- see module docstring",
)
def test_openvino_yolo_pose_inference_runs():
    import cv2
    import numpy as np
    import openvino as ov

    core = ov.Core()
    yolo_model = core.read_model(YOLO_XML)
    yolo_compiled = core.compile_model(yolo_model, "CPU")

    # Dummy frame (640x640)
    padded_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    img = cv2.cvtColor(padded_frame, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1))[np.newaxis, ...].astype(np.float32) / 255.0
    assert img.flags["C_CONTIGUOUS"]

    res = yolo_compiled([img])
    output = list(res.values())[0]

    boxes_transposed = output[0].T  # (8400, 29)
    num_classes = 13
    class_scores = np.max(boxes_transposed[:, 4 : 4 + num_classes], axis=1)

    mask = class_scores > 0.5
    filtered_boxes = boxes_transposed[mask]

    # Smoke-check only -- this dummy random frame has no real markers in it, so
    # the only thing worth asserting is that inference ran and produced
    # well-formed output, not that it found any specific detections.
    assert output.ndim == 3
    assert filtered_boxes.shape[1] == boxes_transposed.shape[1]
