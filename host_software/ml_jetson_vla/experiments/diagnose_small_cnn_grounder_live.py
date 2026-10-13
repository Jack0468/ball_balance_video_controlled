"""One-off diagnostic: calls `SmallCnnGroundingPolicy.act()` directly (synchronous, no threading,
no Policy/_Grounder/_FastWorker machinery) against N real live camera frames, printing the full
`last_debug` dict every frame. Built 2026-10-13 after `run_policy_live.py` showed zero targets for
two different colours (green, black) over ~1075 total ticks, despite `capture_homography.py`
succeeding -- `SmallCnnGroundingPolicy.act()` recomputes ArUco homography fresh on every call (it
never uses a saved --homography-npy), so a one-off capture_homography.py success does not prove
every live frame will also find >=4 markers. This isolates exactly which stage fails: no ArUco,
ArUco-ok-but-no-blob-of-the-requested-colour, or something else -- the live driver's own output
JSON only reports whether a target was ever found, not why not.

Not a replacement for run_policy_live.py -- this never touches serial/motors, it only exercises
the grounder in isolation.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional, Sequence

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ML_JETSON_VLA_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_HOST_SOFTWARE_DIR = os.path.abspath(os.path.join(_ML_JETSON_VLA_DIR, ".."))
_REPO_ROOT_DIR = os.path.abspath(os.path.join(_HOST_SOFTWARE_DIR, ".."))
for _p in (_HOST_SOFTWARE_DIR, _REPO_ROOT_DIR):
    if _p not in sys.path:
        sys.path.append(_p)

from ml_jetson_vla.core.small_cnn_grounder import SmallCnnGroundingPolicy  # noqa: E402


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--instruction", default="go_black", help="e.g. go_red/go_green/go_yellow/go_black")
    parser.add_argument("--n-frames", type=int, default=10)
    parser.add_argument("--save-dir", default=None,
                         help="if set, saves every raw frame as a PNG (raw_frame_<i>.png) for "
                              "visual inspection afterward")
    args = parser.parse_args(argv)

    import cv2

    policy = SmallCnnGroundingPolicy()
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print(f"ERROR: cannot open camera index {args.camera_index}")
        return 1

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    n_aruco_ok = 0
    n_has_target = 0
    try:
        for i in range(args.n_frames):
            ok, frame = cap.read()
            if not ok:
                print(f"frame {i}: camera read failed (ok=False)")
                continue
            if args.save_dir:
                cv2.imwrite(os.path.join(args.save_dir, f"raw_frame_{i}.png"), frame)
            t0 = time.perf_counter()
            command = policy.act(frame, args.instruction, {})
            dt_ms = (time.perf_counter() - t0) * 1000.0
            debug = policy.last_debug
            if debug.get("reason") != "no_aruco":
                n_aruco_ok += 1
            if debug.get("parse_ok"):
                n_has_target += 1
            print(f"frame {i} ({dt_ms:.1f}ms) frame_shape={frame.shape}: {debug}")
    finally:
        cap.release()

    print(f"\nsummary: {n_aruco_ok}/{args.n_frames} frames got past ArUco, "
          f"{n_has_target}/{args.n_frames} found a '{args.instruction}' target")
    return 0


if __name__ == "__main__":
    sys.exit(main())
