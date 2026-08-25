"""probe_camera_modes.py

Diagnostic: measures the REAL achieved frame rate across candidate
(resolution, auto-exposure) combinations for the USB camera. Does not trust
cv2.VideoCapture.get(CAP_PROP_FPS) -- confirmed this session that it just
echoes back whatever was requested via .set(), not what the hardware actually
delivers (requested 60fps, driver reported 60.0 back, but the measured real
inter-frame gap was ~41ms, i.e. ~24fps). Every number this script reports is
a directly-timed inter-frame gap from real cap.read() calls, matching the
methodology already built into USBReceiver's own camera-only instrumentation
(host_software/src/receivers.py).

CAP_PROP_AUTO_EXPOSURE's value convention is backend/driver-dependent and not
reliably documented (0.25 vs 0.75 vs 1 vs 0 all mean different things on
different OpenCV/DirectShow/V4L2 combinations) -- rather than guess which one
this camera uses, this script tries several candidate values and reports both
the readback value (what cap.get() says stuck) and the measured fps for each,
so the actual camera/driver behavior is read off real data, not assumed.

Run directly:
    python host_software/probe_camera_modes.py --cam_id 1
"""

import argparse
import time

import cv2
import numpy as np

RESOLUTIONS = [(320, 240), (640, 480), (800, 600)]

# (label, value-to-set-on-CAP_PROP_AUTO_EXPOSURE). None = leave untouched
# (whatever the camera's power-on default is).
EXPOSURE_CANDIDATES = [
    ("untouched (camera default)", None),
    ("0.75 (DirectShow-manual convention)", 0.75),
    ("0.25 (V4L2-auto-style convention)", 0.25),
    ("1 (manual, alternate convention)", 1.0),
    ("0 (manual, alternate convention)", 0.0),
]

N_WARMUP = 15
N_MEASURE = 60


def probe_one(cam_id, width, height, exposure_label, exposure_value):
    cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        return None

    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, 60)
        if exposure_value is not None:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, exposure_value)

        actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_exposure_flag = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)

        # Warmup: let auto-exposure/auto-focus settle before timing anything.
        # The first real frame after opening/changing settings is routinely a
        # huge outlier (seen this session: 375ms) -- not representative.
        for _ in range(N_WARMUP):
            cap.read()

        gaps_ms = []
        last_t = None
        for _ in range(N_MEASURE):
            t0 = time.perf_counter()
            ret, frame = cap.read()
            t1 = time.perf_counter()
            if not ret or frame is None:
                continue
            if last_t is not None:
                gaps_ms.append((t1 - last_t) * 1000.0)
            last_t = t1

        if not gaps_ms:
            return None

        gaps = np.array(gaps_ms)
        return {
            "requested_res": (width, height),
            "actual_res": (actual_w, actual_h),
            "exposure_label": exposure_label,
            "exposure_readback": actual_exposure_flag,
            "mean_gap_ms": float(gaps.mean()),
            "median_gap_ms": float(np.median(gaps)),
            "p95_gap_ms": float(np.percentile(gaps, 95)),
            "fps": float(1000.0 / gaps.mean()),
        }
    except Exception as e:
        print(f"  ERROR during probe: {e}")
        return None
    finally:
        cap.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe real achievable camera frame rate across resolution/exposure combinations")
    parser.add_argument("--cam_id", type=int, default=1)
    parser.add_argument("--skip-exposure-sweep", action="store_true", help="Only sweep resolutions at the camera's default exposure (faster, skips the 5-way exposure sweep)")
    args = parser.parse_args()

    exposure_candidates = EXPOSURE_CANDIDATES[:1] if args.skip_exposure_sweep else EXPOSURE_CANDIDATES

    results = []
    for width, height in RESOLUTIONS:
        for exposure_label, exposure_value in exposure_candidates:
            print(f"Probing {width}x{height}, exposure={exposure_label} ...")
            r = probe_one(args.cam_id, width, height, exposure_label, exposure_value)
            if r is None:
                print("  FAILED to open/read camera at this mode -- skipping")
                time.sleep(0.5)
                continue
            results.append(r)
            print(
                f"  actual_res={r['actual_res']} exposure_readback={r['exposure_readback']:.3f} | "
                f"mean_gap={r['mean_gap_ms']:.1f}ms median={r['median_gap_ms']:.1f}ms p95={r['p95_gap_ms']:.1f}ms "
                f"-> ~{r['fps']:.1f}fps"
            )
            time.sleep(0.5)  # let the camera fully release before the next open

    if not results:
        print("\nNo successful probes -- camera may be in use by another process, or --cam_id is wrong.")
        return

    print("\n=== Summary (best measured FPS first) ===")
    for r in sorted(results, key=lambda x: -x["fps"]):
        print(
            f"{r['requested_res'][0]}x{r['requested_res'][1]} (actual {r['actual_res'][0]:.0f}x{r['actual_res'][1]:.0f}) "
            f"exposure={r['exposure_label']:<38} -> {r['fps']:6.1f}fps  (mean gap {r['mean_gap_ms']:.1f}ms, p95 {r['p95_gap_ms']:.1f}ms)"
        )

    best = max(results, key=lambda x: x["fps"])
    print(
        f"\nBest: {best['requested_res'][0]}x{best['requested_res'][1]}, exposure={best['exposure_label']} "
        f"-> ~{best['fps']:.1f}fps. If this is meaningfully above ~24fps, pass the matching "
        f"--cam-width/--cam-height/--cam-auto-exposure flags to main_onnx_shared_vision_audio.py."
    )


if __name__ == "__main__":
    main()
