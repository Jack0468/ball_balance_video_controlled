"""Live laptop inference entry point for shared_vision_backbone_v2.

Structural skeleton (audio receiver, TargetStateMachine, PredictionGate,
serial transmission loop) reused wholesale from main_onnx_aruco_audio.py --
that scaffolding is preprocessing-agnostic. What's new/replaced:

  - Preprocessing: perspective warp_to_platform() (auto_label_shared_vision.py)
    instead of an axis-aligned bbox crop -- must match training preprocessing
    exactly (SharedVisionDataset / build_eval_transform), a hard-won lesson
    from this project's train/inference-parity bugs.
  - Model: single multi-head ONNX session (ball_xy, mask_logits, heatmap_logits)
    replacing the old cascaded CNN+MLP.
  - Marker detection: NEW. mask_logits/heatmap_logits -> MarkerClassifier
    (host_software/ml_vision/core/marker_classifier.py) -> state_machine's
    update_markers()/maybe_auto_hold(), which existed but were never fed real
    marker data by any prior entry point.
  - --mlp flag (optional, default off): applies mlp_corrector_shared_vision_v1
    (see train_mlp_corrector_shared_vision.py) on top of the raw CNN output.
    That corrector operates in centered platform mm, NOT the old
    mlp_corrector_time_aruco_0730_v1's raw camera-pixel space -- the two are
    not interchangeable.

See docs/plans (this session's plan file) for the full design rationale.
"""

import argparse
import collections
import os
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort
import serial

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

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
from host_software.ml_vision.core.keyboard_command_receiver import KeyboardCommandReceiver

# --- Configuration ---
SERIAL_PORT = "COM7"
SERIAL_BAUD = 2000000
INPUT_SIZE = (128, 128)  # (H, W), must match training

# ArUco fiducial layout (ids 0-5) is identical across every printed sheet
# (verified this session: ground_truth_manifest.json and
# aruco_markers_01/02/03_manifest.json all have the same aruco_markers list --
# only the colored `features` list differs, and those are no longer needed at
# inference since markers are now detected live by the CNN). So the homography
# lookup can be built once from the base manifest regardless of which sheet is
# physically mounted.
GROUND_TRUTH_MANIFEST = os.path.join(root_dir, "hardware", "platform_templates", "ground_truth_manifest.json")

# Physical platform dimensions -- mm/px conversion is a fixed linear map (not a
# per-frame homography) because warp_to_platform() always warps the same fixed
# mm rectangle onto the full output frame. See
# run_shared_vision_inference_on_dataset.py for the derivation/verification of
# this formula, including the touch_x/touch_y axis-convention fix (the X axis
# is NOT a simple centering subtraction -- see that file's comment).
PAPER_MARGIN_MM = 6.0

# ArUco marker physical positions (mm, centred), used by PredictionGate to
# reject predictions that land on a marker during startup. Same 4 corner
# markers used by main_onnx_aruco_audio.py.
_ARUCO_MARKER_CENTRES_MM_RAW = [
    [12.0, 130.0],
    [175.5, 130.0],
    [175.5, 12.0],
    [12.0, 12.0],
]


class PredictionGate:
    """Two-phase state machine that gates CNN ball predictions -- unchanged
    from main_onnx_aruco_audio.py, since it operates purely on mm floats and
    is model-agnostic."""

    def __init__(
        self,
        marker_centres: np.ndarray,
        marker_radius_mm: float = 20.0,
        jump_threshold_mm: float = 30.0,
        ema_alpha: float = 0.15,
        seed_window: int = 5,
        seed_consistency_mm: float = 15.0,
        lost_frames_threshold: int = 30,
    ) -> None:
        self.marker_centres = marker_centres
        self.marker_radius_mm = marker_radius_mm
        self.jump_threshold_mm = jump_threshold_mm
        self.ema_alpha = ema_alpha
        self.seed_window = seed_window
        self.seed_consistency_mm = seed_consistency_mm
        self.lost_frames_threshold = lost_frames_threshold

        self._phase: str = "AWAITING_BALL"
        self._seed_buffer: list = []
        self._ema = None
        self._last_good = None
        self._consecutive_jumps: int = 0

    @property
    def ball_on_platform(self) -> bool:
        return self._phase == "TRACKING"

    def filter(self, x_mm: float, y_mm: float):
        candidate = np.array([x_mm, y_mm], dtype=np.float32)
        if self._phase == "AWAITING_BALL":
            return self._handle_awaiting(candidate)
        return self._handle_tracking(candidate)

    def _handle_awaiting(self, candidate: np.ndarray):
        if self.marker_centres.shape[0] > 0 and self.marker_radius_mm > 0:
            dists = np.linalg.norm(self.marker_centres - candidate, axis=1)
            if dists.min() < self.marker_radius_mm:
                self._seed_buffer.clear()
                return 0.0, 0.0, "no_ball"
        self._seed_buffer.append(candidate.copy())
        if len(self._seed_buffer) > self.seed_window:
            self._seed_buffer.pop(0)
        if len(self._seed_buffer) == self.seed_window:
            stack = np.stack(self._seed_buffer)
            centroid = stack.mean(axis=0)
            max_dist = float(np.linalg.norm(stack - centroid, axis=1).max())
            if max_dist < self.seed_consistency_mm:
                self._ema = centroid.copy()
                self._last_good = centroid.copy()
                self._phase = "TRACKING"
                self._consecutive_jumps = 0
                self._seed_buffer.clear()
                print(f"\n  ✅ Ball confirmed on platform at ({centroid[0]:+.1f}, {centroid[1]:+.1f}) mm -- tracking started\n")
                return float(centroid[0]), float(centroid[1]), "seeded"
        return 0.0, 0.0, "no_ball"

    def _handle_tracking(self, candidate: np.ndarray):
        if self._ema is not None and self.jump_threshold_mm > 0:
            jump = float(np.linalg.norm(candidate - self._ema))
            if jump > self.jump_threshold_mm:
                self._consecutive_jumps += 1
                if self._consecutive_jumps >= self.lost_frames_threshold:
                    print(f"\n  \U0001f534 Ball lost (>{self.lost_frames_threshold} consecutive jump rejections) -- reverting to AWAITING_BALL\n")
                    self._phase = "AWAITING_BALL"
                    self._ema = None
                    self._last_good = None
                    self._seed_buffer.clear()
                    self._consecutive_jumps = 0
                    return 0.0, 0.0, "no_ball"
                return self._hold("jump_gate")
        self._consecutive_jumps = 0
        if self._ema is None:
            self._ema = candidate.copy()
        else:
            self._ema = self.ema_alpha * candidate + (1.0 - self.ema_alpha) * self._ema
        # last_good stays the raw candidate (jump-gate comparisons must anchor on
        # the true last-observed point, not a lagged average of itself), but the
        # transmitted position uses the smoothed EMA -- previously this returned
        # the raw candidate directly, so every accepted frame's per-frame noise
        # passed to the firmware completely unfiltered. See PROJECT_LOGBOOK.md
        # 18/08/2026 (live-deployment jitter diagnosis).
        self._last_good = candidate.copy()
        return float(self._ema[0]), float(self._ema[1]), "ok"

    def _hold(self, reason: str):
        # Hold at the smoothed EMA, not the raw last-accepted candidate -- keeps
        # the transmitted signal consistently smoothed across accept/reject
        # transitions instead of snapping back to a raw value on every rejection
        # (jump-gate rejections are frequent in practice, per live-deployment logs).
        if self._ema is not None:
            return float(self._ema[0]), float(self._ema[1]), reason
        return 0.0, 0.0, "no_ball"


def preprocess_warped(warped_bgr: np.ndarray) -> np.ndarray:
    """Matches build_eval_transform() + SharedVisionDataset.__getitem__ exactly:
    RGB, [0,1] float, CHW, batch dim. No ImageNet mean/std normalization --
    that was the OLD model's convention, not this one's."""
    rgb = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2RGB)
    img_f = rgb.astype(np.float32) / 255.0
    chw = np.transpose(img_f, (2, 0, 1))
    return np.expand_dims(chw, axis=0)


def px_to_touch_mm(px_x: float, px_y: float, mm_per_px_x: float, mm_per_px_y: float, w_mm: float, h_mm: float):
    manifest_mm_x = -PAPER_MARGIN_MM + px_x * mm_per_px_x
    manifest_mm_y = -PAPER_MARGIN_MM + px_y * mm_per_px_y
    touch_x = w_mm / 2.0 - manifest_mm_x
    touch_y = manifest_mm_y - h_mm / 2.0
    return touch_x, touch_y


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def main() -> None:
    parser = argparse.ArgumentParser(description="shared_vision_backbone_v2 -> CNN + Marker Detection + Audio Tracker (ONNX)")
    parser.add_argument("--cam_id", type=int, default=1, help="Camera ID for USB mode")
    parser.add_argument("--port", type=str, default="auto", help="STM32 serial port or 'auto'")
    parser.add_argument("--udp", action="store_true", help="Use UDP receiver")
    parser.add_argument("--udp_port", type=int, default=5001, help="UDP listen port")
    parser.add_argument("--headless", action="store_true", help="Disable GUI display (improves performance)")
    parser.add_argument("--marker-gate-mm", type=float, default=20.0, help="Radius (mm) around ArUco marker centres to reject predictions during startup (0 to disable)")
    parser.add_argument("--jump-gate-mm", type=float, default=30.0, help="Reject predictions that jump more than this distance (mm) from EMA in one frame (0 to disable)")
    parser.add_argument("--gate-ema-alpha", type=float, default=0.15, help="EMA smoothing factor for the jump gate")
    parser.add_argument("--seed-window", type=int, default=5, help="Consecutive consistent frames required to confirm ball on platform")
    parser.add_argument("--seed-consistency-mm", type=float, default=15.0, help="Max spread (mm) across seed-window frames to be considered consistent")
    parser.add_argument("--lost-frames", type=int, default=30, help="Consecutive jump-rejected frames before ball is considered lost")
    parser.add_argument("--mask-threshold", type=float, default=0.5, help="Sigmoid threshold for marker segmentation mask")
    parser.add_argument("--mlp", action="store_true", help="Apply the shared-vision MLP time corrector on top of raw CNN output (default off -- see PROJECT_LOGBOOK for validation results)")
    parser.add_argument("--mlp-model-path", type=str, default=None, help="Override path to the MLP corrector .pth (default: models/mlp_corrector_shared_vision_v1/mlp_corrector_best.pth)")
    parser.add_argument("--dummy-audio", action="store_true", help="Use typed keyboard commands instead of the trained audio model -- for testing state-machine/target-switching without a working mic (temporary testing aid, see keyboard_command_receiver.py)")
    parser.add_argument("--verbose", action="store_true", help="Print status every frame instead of once per second")
    args = parser.parse_args()

    if args.port == "auto":
        detected_port = find_stm32_port()
        if detected_port:
            print(f"Auto-detected STM32 on {detected_port}")
            args.port = detected_port
        else:
            args.port = SERIAL_PORT
            print(f"Could not auto-detect STM32. Defaulting to {args.port}")

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # ---- 1. Model Init (ONNX) ----
    cnn_path = os.path.abspath(os.path.join(script_dir, "ml_vision/models/shared_vision_backbone_v2/shared_vision_backbone_best.onnx"))
    if not os.path.exists(cnn_path):
        print(f"Error: ONNX model not found at {cnn_path}")
        return

    if not args.dummy_audio:
        audio_path = os.path.abspath(os.path.join(script_dir, "ml_audio/models/audio_command_classifier_v3.onnx"))
        if not os.path.exists(audio_path):
            print(f"Error: ONNX Audio model not found at {audio_path}")
            return

    print("Loading ONNX sessions...")
    cnn_opts = ort.SessionOptions()
    cnn_opts.intra_op_num_threads = 2
    cnn_opts.inter_op_num_threads = 1
    cnn_session = ort.InferenceSession(cnn_path, sess_options=cnn_opts, providers=["CPUExecutionProvider"])
    cnn_input_name = cnn_session.get_inputs()[0].name

    mlp_session = None
    mlp_window = None
    if args.mlp:
        import torch

        from host_software.ml_vision.training.train_mlp_corrector_shared_vision import (
            TOUCHPAD_HALF_H_MM,
            TOUCHPAD_HALF_W_MM,
            MLPCorrectorSharedVision,
        )

        mlp_path = args.mlp_model_path or os.path.abspath(
            os.path.join(script_dir, "ml_vision/models/mlp_corrector_shared_vision_v1/mlp_corrector_best.pth")
        )
        if not os.path.exists(mlp_path):
            print(f"Error: --mlp was passed but no checkpoint found at {mlp_path}.")
            print("Run train_mlp_corrector_shared_vision.py first, or pass --mlp-model-path.")
            return

        # Window size is inferred from the checkpoint's own first-layer weight
        # shape, not taken from a CLI flag -- different mlp_corrector_shared_vision_*
        # variants (e.g. v1 = window 5, vw1 = window 1) have different window
        # sizes baked in, and a mismatched flag produces a state_dict size-mismatch
        # crash (seen live: vw1 loaded with the default window-5 construction).
        # Trusting the checkpoint's actual shape removes that whole failure mode.
        mlp_num_features = 5
        mlp_state_dict = torch.load(mlp_path, map_location="cpu")
        mlp_input_dim = mlp_state_dict["net.0.weight"].shape[1]
        if mlp_input_dim % mlp_num_features != 0:
            print(f"Error: checkpoint at {mlp_path} has input dim {mlp_input_dim}, not a multiple of {mlp_num_features} features -- can't infer window size.")
            return
        mlp_window_size = mlp_input_dim // mlp_num_features

        mlp_model = MLPCorrectorSharedVision(window_size=mlp_window_size)
        mlp_model.load_state_dict(mlp_state_dict)
        mlp_model.eval()
        mlp_session = mlp_model
        mlp_window = collections.deque(maxlen=mlp_window_size)
        print(f"Loaded MLP corrector from {mlp_path} (window_size={mlp_window_size}, auto-detected from checkpoint)")

    if args.dummy_audio:
        audio_receiver = KeyboardCommandReceiver()
    else:
        print("Initializing Audio Receiver...")
        audio_receiver = AudioCommandReceiverONNX(audio_path)
    state_machine = TargetStateMachine()
    marker_classifier = MarkerClassifier(input_size=INPUT_SIZE, mask_threshold=args.mask_threshold)

    # ---- 2. ArUco / Warp Init ----
    aruco_markers, _features, platform_w_mm, platform_h_mm = load_manifest_full(GROUND_TRUTH_MANIFEST)
    aruco_lookup = {int(m["id"]): list(m["center_mm"]) for m in aruco_markers}
    paper_corners = build_paper_corners(platform_w_mm, platform_h_mm, margin_mm=PAPER_MARGIN_MM)
    mm_per_px_x = (platform_w_mm + 2 * PAPER_MARGIN_MM) / INPUT_SIZE[1]
    mm_per_px_y = (platform_h_mm + 2 * PAPER_MARGIN_MM) / INPUT_SIZE[0]

    _ARUCO_CENTRES_CENTRED = np.array(
        [[x - platform_w_mm / 2.0, y - platform_h_mm / 2.0] for x, y in _ARUCO_MARKER_CENTRES_MM_RAW],
        dtype=np.float32,
    )

    # ---- 3. Serial Port Init ----
    try:
        ser = serial.Serial(args.port, SERIAL_BAUD, timeout=0)
        print(f"Connected to STM32 on {args.port} at {SERIAL_BAUD} baud.")
    except Exception:
        print(f"Could not open serial port {args.port}. Continuing in dry-run mode.")
        ser = None

    # ---- 4. Camera/Receiver Init ----
    if args.udp:
        receiver = UDPReceiver(port=args.udp_port, width=640, height=480)
    else:
        receiver = USBReceiver(camera_id=args.cam_id)

    print("Waiting for camera feed...")
    frame = None
    while frame is None:
        frame = receiver.get_latest_frame()
        time.sleep(0.1)

    print("Starting shared_vision_backbone_v2 -> Marker Detection + Audio Tracker loop... (press Ctrl+C to quit)")

    gate = PredictionGate(
        marker_centres=_ARUCO_CENTRES_CENTRED,
        marker_radius_mm=args.marker_gate_mm,
        jump_threshold_mm=args.jump_gate_mm,
        ema_alpha=args.gate_ema_alpha,
        seed_window=args.seed_window,
        seed_consistency_mm=args.seed_consistency_mm,
        lost_frames_threshold=args.lost_frames,
    )
    print(
        f"PredictionGate: marker_gate={args.marker_gate_mm:.0f}mm, jump_gate={args.jump_gate_mm:.0f}mm, "
        f"seed_window={args.seed_window} frames @ {args.seed_consistency_mm:.0f}mm, lost_threshold={args.lost_frames} frames"
    )
    print("  Waiting for ball to be placed on platform...")

    last_status_t: float = 0.0
    last_frame_time = time.perf_counter()

    try:
        while True:
            frame = receiver.get_latest_frame()
            if frame is None:
                continue

            start_t = time.perf_counter()
            dt_ms = (start_t - last_frame_time) * 1000.0
            last_frame_time = start_t
            if dt_ms > 100.0:
                dt_ms = 33.0

            # --- STAGE 1: ArUco Homography + Perspective Warp ---
            aruco_t0 = time.perf_counter()
            aruco_homography = estimate_homography_from_aruco(frame, aruco_lookup)
            aruco_ms = (time.perf_counter() - aruco_t0) * 1000.0

            if aruco_homography is None:
                print(f"Gate closed - Insufficient ArUco markers ({aruco_ms:.1f}ms)")
                continue

            warped_bgr, _warp_matrix = warp_to_platform(frame, aruco_homography, output_size=INPUT_SIZE[::-1], paper_corners=paper_corners)

            # --- STAGE 2: CNN Inference (ball + mask + heatmap) ---
            cnn_t0 = time.perf_counter()
            input_tensor = preprocess_warped(warped_bgr)
            ball_xy, mask_logits, heatmap_logits = cnn_session.run(
                ["ball_xy", "mask_logits", "heatmap_logits"], {cnn_input_name: input_tensor}
            )
            cnn_ms = (time.perf_counter() - cnn_t0) * 1000.0

            norm_x, norm_y = ball_xy[0]
            px_x = float(norm_x) * INPUT_SIZE[1]
            px_y = float(norm_y) * INPUT_SIZE[0]
            raw_x, raw_y = px_to_touch_mm(px_x, px_y, mm_per_px_x, mm_per_px_y, platform_w_mm, platform_h_mm)

            # --- STAGE 3: Marker Detection ---
            mask_prob = sigmoid(mask_logits[0, 0])
            heatmap_prob = sigmoid(heatmap_logits[0, 0])
            detections = marker_classifier.classify(warped_bgr, mask_prob, heatmap_prob)
            marker_coords = {}
            for det in detections:
                if det.color not in marker_coords or det.area_px > marker_coords[det.color][2]:
                    marker_coords[det.color] = (det.x_mm, det.y_mm, det.area_px)
            marker_coords_xy = {color: (x, y) for color, (x, y, _area) in marker_coords.items()}

            # --- STAGE 4: Optional MLP Time Corrector ---
            mlp_ms = 0.0
            if mlp_session is not None:
                mlp_t0 = time.perf_counter()
                mlp_window.append([raw_x / TOUCHPAD_HALF_W_MM, raw_y / TOUCHPAD_HALF_H_MM, 0.0, 0.0, (dt_ms / 33.0) - 1.0])
                if len(mlp_window) == mlp_window.maxlen:
                    feat = torch.tensor(np.array(mlp_window, dtype=np.float32).flatten()).unsqueeze(0)
                    with torch.no_grad():
                        out = mlp_session(feat)[0]
                    final_x, final_y = float(out[0]), float(out[1])
                else:
                    final_x, final_y = raw_x, raw_y
                mlp_ms = (time.perf_counter() - mlp_t0) * 1000.0
            else:
                final_x, final_y = raw_x, raw_y

            # --- STAGE 5: Prediction Gate ---
            gated_x, gated_y, gate_reason = gate.filter(final_x, final_y)

            if gate_reason == "no_ball":
                command = audio_receiver.get_latest_command()
                if command:
                    print(f"\n[AUDIO] (gate=no_ball) Heard: {command} -- waiting for ball\n")
                continue

            if gate_reason not in ("ok", "seeded"):
                print(f"  ⚠ Gate [{gate_reason}] rejected ({final_x:+.1f}, {final_y:+.1f}) mm -> holding ({gated_x:+.1f}, {gated_y:+.1f}) mm")
            final_x, final_y = gated_x, gated_y

            # --- STAGE 6: Audio + State Machine ---
            command = audio_receiver.get_latest_command()
            if command:
                print(f"\n[AUDIO] Heard command: {command}\n")

            state_machine.process_command(command, final_x, final_y)
            state_machine.update_markers(marker_coords_xy)
            state_machine.maybe_auto_hold(final_x, final_y, marker_coords_xy)
            target_x, target_y = state_machine.get_target_coords()

            # --- STAGE 7: Serial Transmission ---
            try:
                payload = f"{final_x:.2f},{final_y:.2f},{target_x:.2f},{target_y:.2f}\n".encode("ascii")
                if ser is not None:
                    ser.write(payload)
            except Exception as e:
                print(f"Serial Error: {e}")

            end_t = time.perf_counter()
            total_ms = (end_t - start_t) * 1000.0
            fps = 1.0 / (end_t - start_t)
            audio_ms = getattr(audio_receiver, "latest_inference_time_ms", 0.0)
            phase = "TRACKING" if gate.ball_on_platform else "AWAITING_BALL"

            if args.verbose or (end_t - last_status_t) >= 1.0:
                markers_str = ", ".join(f"{c}=({x:+.0f},{y:+.0f})" for c, (x, y) in marker_coords_xy.items()) or "none"
                print(
                    f"[{phase}] Ball: X={final_x:+6.1f} Y={final_y:+6.1f} mm | "
                    f"Target: {state_machine.current_target_name} at X={target_x:.1f} Y={target_y:.1f} | "
                    f"Markers: {markers_str} | Cmd: {state_machine._last_command} | FPS: {fps:.1f} | "
                    f"Total={total_ms:.1f}ms (ArUco={aruco_ms:.1f}ms, CNN={cnn_ms:.1f}ms, MLP={mlp_ms:.1f}ms, Audio={audio_ms:.1f}ms)"
                )
                last_status_t = end_t

            if not args.headless:
                disp = warped_bgr.copy()
                cv2.circle(disp, (int(px_x), int(px_y)), 4, (0, 0, 255), -1)
                for i, det in enumerate(detections):
                    label = f"{det.color}/{det.shape}"
                    cv2.putText(disp, label, (5, 15 + i * 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                cv2.imshow("Shared Vision Backbone Tracker (ONNX)", disp)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        pass
    finally:
        receiver.stop()
        audio_receiver.stop()
        if ser:
            ser.close()
        cv2.destroyAllWindows()
        print("Inference loop stopped.")


if __name__ == "__main__":
    main()
