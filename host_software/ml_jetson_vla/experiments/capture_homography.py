"""One-off utility: capture a live frame from the platform's camera, compute the mm->px ArUco
homography, and save it as a .npy file for `run_experiment.py --mode live --homography-npy`.

Reuses `estimate_homography_from_aruco()` (`ml_vision/data_processing/auto_label_shared_vision.py`)
-- the same function Track 1's `runtime/run_jetson_standalone.py` already uses in production, not a
new computation. Run directly with host python3, NOT through the arm2-lerobot Docker image: this
needs direct access to the camera device and has no torch/lerobot dependency.
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ML_JETSON_VLA_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_HOST_SOFTWARE_DIR = os.path.abspath(os.path.join(_ML_JETSON_VLA_DIR, ".."))
_REPO_ROOT_DIR = os.path.abspath(os.path.join(_HOST_SOFTWARE_DIR, ".."))
for _p in (_HOST_SOFTWARE_DIR, _REPO_ROOT_DIR):
    if _p not in sys.path:
        sys.path.append(_p)

from host_software.ml_vision.data_processing.auto_label_shared_vision import (  # noqa: E402
    estimate_homography_from_aruco,
    load_manifest_full,
)

# Same lightweight path-join pattern as deployment/score_minimal_baseline_offline.py's
# GROUND_TRUTH_MANIFEST -- avoids pulling in main_onnx_shared_vision_audio.py's heavier imports
# (onnxruntime etc.) just for this one constant.
GROUND_TRUTH_MANIFEST: str = os.path.join(
    _REPO_ROOT_DIR, "hardware", "platform_templates", "ground_truth_manifest.json"
)


def capture_homography(camera_index: int, n_attempts: int) -> np.ndarray:
    """Returns the mm->px homography from the first frame with >=4 detected ArUco markers."""
    aruco_markers, _features, _platform_w_mm, _platform_h_mm = load_manifest_full(GROUND_TRUTH_MANIFEST)
    aruco_lookup = {int(m["id"]): list(m["center_mm"]) for m in aruco_markers}

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera {camera_index}")
    try:
        for attempt in range(n_attempts):
            ok, frame = cap.read()
            if not ok:
                continue
            homography = estimate_homography_from_aruco(frame, aruco_lookup)
            if homography is not None:
                return homography
        raise RuntimeError(
            f"could not detect >=4 ArUco markers in {n_attempts} frames from camera {camera_index} "
            "-- check framing, lighting and focus, and that the platform's markers are unoccluded"
        )
    finally:
        cap.release()


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--out", required=True, help="output .npy path (a .npy extension is added if missing)")
    parser.add_argument("--n-attempts", type=int, default=10, help="frames to try before giving up")
    args = parser.parse_args(argv)
    if args.n_attempts < 1:
        parser.error("--n-attempts must be >= 1")

    homography = capture_homography(args.camera_index, args.n_attempts)

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, homography)
    print(f"saved mm->px homography {homography.shape} to {out_path}")
    print(homography)
    return 0


if __name__ == "__main__":
    sys.exit(main())
