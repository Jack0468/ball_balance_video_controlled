"""Jetson entry point for the small-class expert pipeline (Track 1 of the Jetson port
plan, 2026-08-18). Runtime-target port of `host_software/main_onnx_shared_vision_audio.py`
-- same pipeline (ArUco homography+warp -> shared_vision_backbone_v2 CNN -> marker
detection -> audio ONNX -> state machine -> PredictionGate), same wire protocol. Reuses
that script's `PredictionGate`, `preprocess_warped`, `px_to_touch_mm`, `sigmoid`, and
manifest/margin constants directly (import, not copy-paste) since none of that logic is
platform-specific -- only what changes below is:

  - Camera open: unchanged code path (`src.receivers.USBReceiver`), but on Linux/Jetson
    its `cv2.CAP_DSHOW` attempt fails and it falls through to the default backend (V4L2).
    That fallback already existed for other reasons; this is the first place it's expected
    to actually engage. Verify this on real hardware -- don't assume.
  - Serial port default: Windows' "COM7" fallback replaced with a Linux tty path. Real
    detection still goes through `find_stm32_port()` (`src.utils`), which is already
    OS-agnostic (matches on `pyserial` port description, not a Windows-specific string).
  - ONNX providers: CPUExecutionProvider only, matching this session's CPU-first decision
    for the small-class pipeline (~70-91K/13.5K params -- already real-time on a laptop
    CPU, no GPU work justified here). See Track 3 for where GPU/TensorRT actually matters.

**Phase A only**: the control net still runs on the STM32 (`RLControl.cpp`), unchanged.
This script sends `ball_x,ball_y,target_x,target_y` ASCII over `SerialCoords.cpp` exactly
like the laptop version -- it does NOT use `ml_jetson_vla/core/control_net.py` or
`RemoteStepControl.cpp`. Migrating control-net inference onto the Jetson is Phase B,
deferred pending the firmware telemetry-back work proposed in `../stm32_interface/`.

Logic is wrapped behind `core.policy_interface.Policy` (`JetsonExpertPolicy` below) so a
future arm (Track 3's medium class, or a large-VLA arm) can be swapped in without touching
the camera/serial/runtime-loop plumbing in `main()`.
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort
import serial

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_HOST_SOFTWARE_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_REPO_ROOT_DIR = os.path.abspath(os.path.join(_HOST_SOFTWARE_DIR, ".."))
for _p in (_HOST_SOFTWARE_DIR, _REPO_ROOT_DIR):
    if _p not in sys.path:
        sys.path.append(_p)

from src.receivers import USBReceiver, UDPReceiver
from src.utils import find_stm32_port
from src.state_machine import TargetStateMachine
from src.audio_receiver_onnx import AudioCommandReceiverONNX

from host_software.ml_vision.data_processing.auto_label_shared_vision import (
    build_paper_corners,
    estimate_homography_from_aruco,
    load_manifest_full,
    warp_to_platform,
)
from host_software.ml_vision.core.marker_classifier import MarkerClassifier

# Reused as-is from the laptop entry point -- this logic is preprocessing/warp/gating
# math with no platform dependency. Importing (not duplicating) keeps the two entry
# points from silently drifting apart.
from main_onnx_shared_vision_audio import (
    PredictionGate,
    preprocess_warped,
    px_to_touch_mm,
    sigmoid,
    PAPER_MARGIN_MM,
    GROUND_TRUTH_MANIFEST,
    INPUT_SIZE,
    _ARUCO_MARKER_CENTRES_MM_RAW,
)

from ml_jetson_vla.core.policy_interface import Policy, PolicyCommand

# --- Configuration ---
# Laptop default was "COM7". No physical default tty is more "correct" than another on
# Linux -- this is just a fallback for when find_stm32_port() (VID/PID + description
# matching, already OS-agnostic) comes up empty, not something to rely on normally.
SERIAL_PORT = "/dev/ttyACM0"
SERIAL_BAUD = 2000000


class JetsonExpertPolicy(Policy):
    """Small-class expert pipeline (vision CNN + marker detection + audio + state
    machine), Phase A: computes a target in mm, does NOT compute step targets -- the
    STM32's own `RLControl.cpp` still does that. Owns the CNN/audio ONNX sessions, the
    marker classifier, the state machine, and the prediction gate; `act()` is one frame."""

    def __init__(self, script_dir: str, marker_gate_mm: float, jump_gate_mm: float,
                 gate_ema_alpha: float, seed_window: int, seed_consistency_mm: float,
                 lost_frames: int, mask_threshold: float) -> None:
        cnn_path = os.path.abspath(
            os.path.join(script_dir, "ml_vision/models/shared_vision_backbone_v2/shared_vision_backbone_best.onnx")
        )
        if not os.path.exists(cnn_path):
            raise FileNotFoundError(f"ONNX vision model not found at {cnn_path}")
        audio_path = os.path.abspath(
            os.path.join(script_dir, "ml_audio/models/audio_command_classifier_v3.onnx")
        )
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"ONNX audio model not found at {audio_path}")

        print("Loading ONNX sessions (CPUExecutionProvider)...")
        cnn_opts = ort.SessionOptions()
        cnn_opts.intra_op_num_threads = 2
        cnn_opts.inter_op_num_threads = 1
        self.cnn_session = ort.InferenceSession(cnn_path, sess_options=cnn_opts, providers=["CPUExecutionProvider"])
        self.cnn_input_name = self.cnn_session.get_inputs()[0].name

        self.audio_receiver = AudioCommandReceiverONNX(audio_path)
        self.state_machine = TargetStateMachine()
        self.marker_classifier = MarkerClassifier(input_size=INPUT_SIZE, mask_threshold=mask_threshold)

        aruco_markers, _features, platform_w_mm, platform_h_mm = load_manifest_full(GROUND_TRUTH_MANIFEST)
        self.aruco_lookup = {int(m["id"]): list(m["center_mm"]) for m in aruco_markers}
        self.paper_corners = build_paper_corners(platform_w_mm, platform_h_mm, margin_mm=PAPER_MARGIN_MM)
        self.mm_per_px_x = (platform_w_mm + 2 * PAPER_MARGIN_MM) / INPUT_SIZE[1]
        self.mm_per_px_y = (platform_h_mm + 2 * PAPER_MARGIN_MM) / INPUT_SIZE[0]
        self.platform_w_mm = platform_w_mm
        self.platform_h_mm = platform_h_mm

        aruco_centres_centred = np.array(
            [[x - platform_w_mm / 2.0, y - platform_h_mm / 2.0] for x, y in _ARUCO_MARKER_CENTRES_MM_RAW],
            dtype=np.float32,
        )
        self.gate = PredictionGate(
            marker_centres=aruco_centres_centred,
            marker_radius_mm=marker_gate_mm,
            jump_threshold_mm=jump_gate_mm,
            ema_alpha=gate_ema_alpha,
            seed_window=seed_window,
            seed_consistency_mm=seed_consistency_mm,
            lost_frames_threshold=lost_frames,
        )
        self.last_debug: dict = {}

    def reset(self) -> None:
        self.gate = PredictionGate(
            marker_centres=self.gate.marker_centres,
            marker_radius_mm=self.gate.marker_radius_mm,
            jump_threshold_mm=self.gate.jump_threshold_mm,
            ema_alpha=self.gate.ema_alpha,
            seed_window=self.gate.seed_window,
            seed_consistency_mm=self.gate.seed_consistency_mm,
            lost_frames_threshold=self.gate.lost_frames_threshold,
        )

    def act(self, image: np.ndarray, instruction, state: dict) -> "PolicyCommand | None":
        aruco_homography = estimate_homography_from_aruco(image, self.aruco_lookup)
        if aruco_homography is None:
            self.last_debug = {"reason": "no_aruco"}
            return None

        warped_bgr, _warp_matrix = warp_to_platform(
            image, aruco_homography, output_size=INPUT_SIZE[::-1], paper_corners=self.paper_corners
        )

        input_tensor = preprocess_warped(warped_bgr)
        ball_xy, mask_logits, heatmap_logits = self.cnn_session.run(
            ["ball_xy", "mask_logits", "heatmap_logits"], {self.cnn_input_name: input_tensor}
        )
        norm_x, norm_y = ball_xy[0]
        px_x = float(norm_x) * INPUT_SIZE[1]
        px_y = float(norm_y) * INPUT_SIZE[0]
        raw_x, raw_y = px_to_touch_mm(px_x, px_y, self.mm_per_px_x, self.mm_per_px_y, self.platform_w_mm, self.platform_h_mm)

        mask_prob = sigmoid(mask_logits[0, 0])
        heatmap_prob = sigmoid(heatmap_logits[0, 0])
        detections = self.marker_classifier.classify(warped_bgr, mask_prob, heatmap_prob)
        marker_coords: dict = {}
        for det in detections:
            if det.color not in marker_coords or det.area_px > marker_coords[det.color][2]:
                marker_coords[det.color] = (det.x_mm, det.y_mm, det.area_px)
        marker_coords_xy = {color: (x, y) for color, (x, y, _area) in marker_coords.items()}

        gated_x, gated_y, gate_reason = self.gate.filter(raw_x, raw_y)
        self.last_debug = {"gate_reason": gate_reason, "warped_bgr": warped_bgr, "detections": detections}
        if gate_reason == "no_ball":
            return None

        command = instruction
        if command:
            self.state_machine.process_command(command, gated_x, gated_y)
        self.state_machine.update_markers(marker_coords_xy)
        self.state_machine.maybe_auto_hold(gated_x, gated_y, marker_coords_xy)
        target_x, target_y = self.state_machine.get_target_coords()

        self.last_debug["ball_xy_mm"] = (gated_x, gated_y)
        self.last_debug["markers"] = marker_coords_xy
        return PolicyCommand(target_x_mm=target_x, target_y_mm=target_y)


def main() -> None:
    parser = argparse.ArgumentParser(description="Jetson standalone entry point -- small-class expert pipeline (Phase A)")
    parser.add_argument("--cam_id", type=int, default=0, help="Camera device index (e.g. /dev/video0 -> 0)")
    parser.add_argument("--port", type=str, default="auto", help="STM32 serial port or 'auto'")
    parser.add_argument("--udp", action="store_true", help="Use UDP receiver instead of a local USB camera")
    parser.add_argument("--udp_port", type=int, default=5001)
    parser.add_argument("--headless", action="store_true", help="Disable GUI display (recommended over SSH)")
    parser.add_argument("--marker-gate-mm", type=float, default=20.0)
    parser.add_argument("--jump-gate-mm", type=float, default=30.0)
    parser.add_argument("--gate-ema-alpha", type=float, default=0.15)
    parser.add_argument("--seed-window", type=int, default=5)
    parser.add_argument("--seed-consistency-mm", type=float, default=15.0)
    parser.add_argument("--lost-frames", type=int, default=30)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.port == "auto":
        detected_port = find_stm32_port()
        if detected_port:
            print(f"Auto-detected STM32 on {detected_port}")
            args.port = detected_port
        else:
            args.port = SERIAL_PORT
            print(f"Could not auto-detect STM32. Defaulting to {args.port}")

    policy = JetsonExpertPolicy(
        script_dir=_HOST_SOFTWARE_DIR,
        marker_gate_mm=args.marker_gate_mm,
        jump_gate_mm=args.jump_gate_mm,
        gate_ema_alpha=args.gate_ema_alpha,
        seed_window=args.seed_window,
        seed_consistency_mm=args.seed_consistency_mm,
        lost_frames=args.lost_frames,
        mask_threshold=args.mask_threshold,
    )

    try:
        ser = serial.Serial(args.port, SERIAL_BAUD, timeout=0)
        print(f"Connected to STM32 on {args.port} at {SERIAL_BAUD} baud.")
    except Exception:
        print(f"Could not open serial port {args.port}. Continuing in dry-run mode.")
        ser = None

    if args.udp:
        receiver = UDPReceiver(port=args.udp_port, width=640, height=480)
    else:
        receiver = USBReceiver(camera_id=args.cam_id)

    print("Waiting for camera feed...")
    frame = None
    while frame is None:
        frame = receiver.get_latest_frame()
        time.sleep(0.1)

    print("Starting Jetson standalone loop (Track 1, Phase A -- STM32 keeps its own control net)... Ctrl+C to quit")
    last_status_t = 0.0

    try:
        while True:
            frame = receiver.get_latest_frame()
            if frame is None:
                continue

            start_t = time.perf_counter()
            command = policy.audio_receiver.get_latest_command()
            cmd_out = policy.act(frame, command, state={})

            if cmd_out is None:
                if command:
                    print(f"\n[AUDIO] (no ball) Heard: {command}\n")
                continue

            try:
                payload = f"{policy.last_debug['ball_xy_mm'][0]:.2f},{policy.last_debug['ball_xy_mm'][1]:.2f},{cmd_out.target_x_mm:.2f},{cmd_out.target_y_mm:.2f}\n".encode("ascii")
                if ser is not None:
                    ser.write(payload)
            except Exception as e:
                print(f"Serial Error: {e}")

            total_ms = (time.perf_counter() - start_t) * 1000.0
            now = time.perf_counter()
            if args.verbose or (now - last_status_t) >= 1.0:
                bx, by = policy.last_debug["ball_xy_mm"]
                markers_str = ", ".join(f"{c}=({x:+.0f},{y:+.0f})" for c, (x, y) in policy.last_debug["markers"].items()) or "none"
                print(
                    f"[{policy.last_debug['gate_reason']}] Ball: X={bx:+6.1f} Y={by:+6.1f} mm | "
                    f"Target: X={cmd_out.target_x_mm:.1f} Y={cmd_out.target_y_mm:.1f} | "
                    f"Markers: {markers_str} | Frame={total_ms:.1f}ms"
                )
                last_status_t = now

            if not args.headless:
                disp = policy.last_debug["warped_bgr"].copy()
                cv2.imshow("Jetson Standalone (Track 1)", disp)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        pass
    finally:
        receiver.stop()
        policy.audio_receiver.stop()
        if ser:
            ser.close()
        cv2.destroyAllWindows()
        print("Jetson standalone loop stopped.")


if __name__ == "__main__":
    main()
