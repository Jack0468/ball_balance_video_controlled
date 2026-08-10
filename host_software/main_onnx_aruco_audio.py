import cv2
import time
import numpy as np
import os
import sys
import serial
import argparse
import collections
import cv2.aruco as aruco
import onnxruntime as ort

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from src.receivers import USBReceiver, UDPReceiver
from src.utils import find_stm32_port
from src.state_machine import TargetStateMachine
from src.audio_receiver_onnx import AudioCommandReceiverONNX

# --- Configuration ---
SERIAL_PORT = "COM7"
SERIAL_BAUD = 2000000
CROP_PAD = 20  # Pixels of padding around the platform crop

# True physical plate bounds
PLATFORM_W = 187.5
PLATFORM_H = 142.0

# These are the exact millimeter coordinates of the centers of the 6 markers
# relative to the Top-Left (0,0) corner of the printed PDF bounding box.
MARKER_PHYSICAL_MM = {
    0: [12.0, 130.0],
    1: [175.5, 130.0],
    2: [175.5, 12.0],
    3: [12.0, 12.0],
    4: [12.0, 71.0],
    5: [175.5, 71.0],
}

# The 4 physical corners of the platform boundary
PLATFORM_CORNERS_MM = np.array(
    [
        [[0.0, 0.0]],
        [[PLATFORM_W, 0.0]],
        [[PLATFORM_W, PLATFORM_H]],
        [[0.0, PLATFORM_H]],
    ],
    dtype=np.float32,
)
# ---------------------

# ArUco marker physical positions (mm, centred: origin at platform centre).
# Used by PredictionGate to reject predictions that land on a marker during startup.
_ARUCO_MARKER_CENTRES_MM_RAW = [
    [12.0, 130.0],   # marker 0
    [175.5, 130.0],  # marker 1
    [175.5, 12.0],   # marker 2
    [12.0, 12.0],    # marker 3
]
_ARUCO_CENTRES_CENTRED = np.array(
    [
        [x - PLATFORM_W / 2.0, y - PLATFORM_H / 2.0]
        for x, y in _ARUCO_MARKER_CENTRES_MM_RAW
    ],
    dtype=np.float32,
)


class PredictionGate:
    """Two-phase state machine that gates CNN+MLP ball predictions.

    Phase AWAITING_BALL
    -------------------
    Active on startup and whenever the ball is removed from the platform.
    - Marker proximity gate is ACTIVE: any prediction within `marker_radius_mm`
      of a known ArUco centre is rejected (ball never starts on a marker).
    - Predictions are accumulated in a sliding window.  Once `seed_window`
      consecutive predictions all lie within `seed_consistency_mm` of their
      centroid the ball is considered confirmed and the gate transitions to
      TRACKING.  The EMA is seeded from the centroid.
    - While in this phase `filter()` returns reason='no_ball' so the caller
      knows NOT to transmit to the firmware.

    Phase TRACKING
    --------------
    Normal operation once the ball is confirmed on the platform.
    - Marker gate is DISABLED: the ball can legitimately roll over any of
      the 6 ArUco markers (only 4 are needed for homography at any time).
    - Jump gate is ACTIVE: predictions that move more than `jump_threshold_mm`
      from the EMA in one frame are rejected and the last good position is
      held.  The EMA is only updated on accepted frames.
    - If `lost_frames_threshold` consecutive frames are rejected by the jump
      gate the gate reverts to AWAITING_BALL (ball removed).
    """

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
        self._seed_buffer: list[np.ndarray] = []
        self._ema: np.ndarray | None = None
        self._last_good: np.ndarray | None = None
        self._consecutive_jumps: int = 0

    @property
    def ball_on_platform(self) -> bool:
        return self._phase == "TRACKING"

    def filter(self, x_mm: float, y_mm: float) -> tuple[float, float, str]:
        candidate = np.array([x_mm, y_mm], dtype=np.float32)
        if self._phase == "AWAITING_BALL":
            return self._handle_awaiting(candidate)
        return self._handle_tracking(candidate)

    def _handle_awaiting(self, candidate: np.ndarray) -> tuple[float, float, str]:
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
                print(
                    f"\n  ✅ Ball confirmed on platform at "
                    f"({centroid[0]:+.1f}, {centroid[1]:+.1f}) mm — tracking started\n"
                )
                return float(centroid[0]), float(centroid[1]), "seeded"
        return 0.0, 0.0, "no_ball"

    def _handle_tracking(self, candidate: np.ndarray) -> tuple[float, float, str]:
        if self._ema is not None and self.jump_threshold_mm > 0:
            jump = float(np.linalg.norm(candidate - self._ema))
            if jump > self.jump_threshold_mm:
                self._consecutive_jumps += 1
                if self._consecutive_jumps >= self.lost_frames_threshold:
                    print(
                        f"\n  🔴 Ball lost (>{self.lost_frames_threshold} consecutive jump "
                        f"rejections) — reverting to AWAITING_BALL\n"
                    )
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
        self._last_good = candidate.copy()
        return float(candidate[0]), float(candidate[1]), "ok"

    def _hold(self, reason: str) -> tuple[float, float, str]:
        if self._last_good is not None:
            return float(self._last_good[0]), float(self._last_good[1]), reason
        return 0.0, 0.0, "no_ball"


def preprocess_numpy(img):
    # cv2 uses (width, height) for resize
    img = cv2.resize(img, (320, 240))
    img = img.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std

    # HWC to CHW
    img = np.transpose(img, (2, 0, 1))

    # Add batch dim -> (1, 3, 240, 320)
    return np.expand_dims(img, axis=0)


def main():
    parser = argparse.ArgumentParser(
        description="Cascaded ArUco -> CNN -> MLP Ball Tracker (ONNX)"
    )
    parser.add_argument("--cam_id", type=int, default=1, help="Camera ID for USB mode")
    parser.add_argument(
        "--port", type=str, default="auto", help="STM32 serial port or 'auto'"
    )
    parser.add_argument("--udp", action="store_true", help="Use UDP receiver")
    parser.add_argument("--udp_port", type=int, default=5001, help="UDP listen port")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Disable GUI display (improves performance)",
    )
    parser.add_argument(
        "--marker-gate-mm",
        type=float,
        default=20.0,
        help="Radius (mm) around ArUco marker centres to reject predictions during startup (0 to disable)",
    )
    parser.add_argument(
        "--jump-gate-mm",
        type=float,
        default=30.0,
        help="Reject predictions that jump more than this distance (mm) from EMA in one frame (0 to disable)",
    )
    parser.add_argument(
        "--gate-ema-alpha",
        type=float,
        default=0.15,
        help="EMA smoothing factor for the jump gate (0=frozen, 1=no smoothing)",
    )
    parser.add_argument(
        "--seed-window",
        type=int,
        default=5,
        help="Consecutive consistent frames required to confirm ball on platform",
    )
    parser.add_argument(
        "--seed-consistency-mm",
        type=float,
        default=15.0,
        help="Max spread (mm) across seed-window frames to be considered consistent",
    )
    parser.add_argument(
        "--lost-frames",
        type=int,
        default=30,
        help="Consecutive jump-rejected frames before ball is considered lost (~1s at 30fps)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print status every frame (default: once per second). Adds ~1-2ms overhead per frame.",
    )
    args = parser.parse_args()

    # Auto-detect serial port
    if args.port == "auto":
        detected_port = find_stm32_port()
        if detected_port:
            print(f"Auto-detected STM32 on {detected_port}")
            args.port = detected_port
        else:
            args.port = SERIAL_PORT
            print(f"Could not auto-detect STM32. Defaulting to {args.port}")

    # ---- 1. Model Init (ONNX) ----
    script_dir = os.path.dirname(os.path.abspath(__file__))

    cnn_path = os.path.abspath(
        os.path.join(
            script_dir, "ml_vision/models/cnn_2d_tracker_0730_v3/expert_tracker_best.onnx"
        )
    )
    mlp_path = os.path.abspath(
        os.path.join(
            script_dir,
            "ml_vision/models/mlp_corrector_time_aruco_0730_v1/mlp_corrector_best.onnx",
        )
    )

    if not os.path.exists(cnn_path):
        print(f"Error: ONNX CNN model not found at {cnn_path}")
        print("Please run export_to_onnx.py first!")
        return

    if not os.path.exists(mlp_path):
        print(f"Error: ONNX MLP model not found at {mlp_path}")
        print("Please run export_to_onnx.py first!")
        return

    audio_path = os.path.abspath(
        os.path.join(script_dir, "ml_audio/models/audio_command_classifier_v3.onnx")
    )
    if not os.path.exists(audio_path):
        print(f"Error: ONNX Audio model not found at {audio_path}")
        print("Please run ml_audio/export_audio_to_onnx.py first!")
        return

    # Initialize ONNX Sessions
    print("Loading ONNX sessions...")

    cnn_opts = ort.SessionOptions()
    cnn_opts.intra_op_num_threads = 2
    cnn_opts.inter_op_num_threads = 1
    cnn_session = ort.InferenceSession(
        cnn_path, sess_options=cnn_opts, providers=["CPUExecutionProvider"]
    )

    mlp_opts = ort.SessionOptions()
    mlp_opts.intra_op_num_threads = 1
    mlp_opts.inter_op_num_threads = 1
    mlp_session = ort.InferenceSession(
        mlp_path, sess_options=mlp_opts, providers=["CPUExecutionProvider"]
    )

    print("Initializing Audio Receiver...")
    audio_receiver = AudioCommandReceiverONNX(audio_path)
    state_machine = TargetStateMachine()

    cnn_input_name = cnn_session.get_inputs()[0].name
    mlp_input_name = mlp_session.get_inputs()[0].name

    # ---- 2. ArUco Init ----
    try:
        dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        parameters = aruco.DetectorParameters()
        detector = aruco.ArucoDetector(dictionary, parameters)
    except AttributeError:
        dictionary = aruco.Dictionary_get(aruco.DICT_4X4_50)
        parameters = aruco.DetectorParameters_create()
        detector = None

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

    print(f"Starting ArUco -> CNN -> MLP + Audio Tracker loop... (press Ctrl+C to quit)")

    # --- Prediction Gate ---
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
        f"PredictionGate: marker_gate={args.marker_gate_mm:.0f}mm, "
        f"jump_gate={args.jump_gate_mm:.0f}mm, "
        f"seed_window={args.seed_window} frames @ {args.seed_consistency_mm:.0f}mm, "
        f"lost_threshold={args.lost_frames} frames"
    )
    print("  Waiting for ball to be placed on platform...")

    # Throttled-print state (used when --verbose is not set)
    last_status_t: float = 0.0

    # Time-Series History Buffer for the MLP
    history_buffer = collections.deque(maxlen=1)
    last_frame_time = time.perf_counter()

    try:
        while True:
            frame = receiver.get_latest_frame()
            if frame is None:
                continue

            start_t = time.perf_counter()
            dt_ms = (start_t - last_frame_time) * 1000.0
            last_frame_time = start_t

            # Bound dt for safety if there's a huge lag spike
            if dt_ms > 100.0:
                dt_ms = 33.0
                history_buffer.clear()  # Clear buffer on large skips

            h_frame, w_frame = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # --- STAGE 1: ArUco Homography ---
            aruco_t0 = time.perf_counter()
            if detector is not None:
                corners, ids, rejected = detector.detectMarkers(gray)
            else:
                corners, ids, rejected = aruco.detectMarkers(
                    gray, dictionary, parameters=parameters
                )

            M = None
            if ids is not None:
                ids = ids.flatten()
                pixel_centers = []
                physical_centers = []

                for i, marker_id in enumerate(ids):
                    if marker_id in MARKER_PHYSICAL_MM:
                        marker_corners = corners[i][0]
                        center = np.mean(marker_corners, axis=0)
                        pixel_centers.append(center)
                        physical_centers.append(MARKER_PHYSICAL_MM[marker_id])

                if len(pixel_centers) >= 4:
                    pixel_centers = np.array(pixel_centers, dtype=np.float32)
                    physical_centers = np.array(physical_centers, dtype=np.float32)
                    M, _ = cv2.findHomography(pixel_centers, physical_centers)

            aruco_ms = (time.perf_counter() - aruco_t0) * 1000.0

            if M is None:
                print(f"Gate closed - Insufficient ArUco markers ({aruco_ms:.1f}ms)")
                continue

            # --- STAGE 2: Crop to Platform ---
            try:
                M_inv = np.linalg.inv(M)
                # Project the 4 physical corners of the board back to pixels
                pixel_corners = cv2.perspectiveTransform(PLATFORM_CORNERS_MM, M_inv)
            except np.linalg.LinAlgError:
                print("Degenerate homography matrix (singular) — skipping")
                continue

            xs = pixel_corners[:, 0, 0]
            ys = pixel_corners[:, 0, 1]
            x1 = int(max(0, xs.min() - CROP_PAD))
            y1 = int(max(0, ys.min() - CROP_PAD))
            x2 = int(min(w_frame, xs.max() + CROP_PAD))
            y2 = int(min(h_frame, ys.max() + CROP_PAD))

            if x2 <= x1 or y2 <= y1:
                print("Degenerate crop — skipping")
                continue

            crop = frame[y1:y2, x1:x2]

            # --- STAGE 3: CNN Ball Tracker (ONNX) ---
            cnn_t0 = time.perf_counter()
            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            input_tensor = preprocess_numpy(rgb_crop)

            # ONNX Inference
            output = cnn_session.run(None, {cnn_input_name: input_tensor})[0]

            cnn_ms = (time.perf_counter() - cnn_t0) * 1000.0

            # The CNN predicts [-1, 1] relative to the crop dimensions!
            norm_x, norm_y = output[0]

            crop_w = x2 - x1
            crop_h = y2 - y1

            # Convert [-1, 1] back to [0, crop_w] and [0, crop_h]
            ball_crop_x = (norm_x + 1.0) * (crop_w / 2.0)
            ball_crop_y = (norm_y + 1.0) * (crop_h / 2.0)

            # Convert crop pixels to full frame pixels
            ball_frame_x = ball_crop_x + x1
            ball_frame_y = ball_crop_y + y1

            # Use Homography (M) to map frame pixels to physical platform mm!
            ball_pt = np.array([[[ball_frame_x, ball_frame_y]]], dtype=np.float32)
            touch_pt = cv2.perspectiveTransform(ball_pt, M)

            # touch_x and touch_y are now perfectly in platform millimeters (0 to PLATFORM_W)
            touch_x_raw = float(touch_pt[0, 0, 0])
            touch_y_raw = float(touch_pt[0, 0, 1])

            # We must center the target relative to the PID firmware!
            centered_touch_x = touch_x_raw - (PLATFORM_W / 2.0)
            centered_touch_y = touch_y_raw - (PLATFORM_H / 2.0)

            # --- STAGE 4: MLP Time Corrector (ONNX) ---
            mlp_t0 = time.perf_counter()

            # Normalize inputs matching the V2 MLP training!
            norm_cnn_x = (ball_frame_x / 320.0) - 1.0
            norm_cnn_y = (ball_frame_y / 240.0) - 1.0
            norm_target_x = 0.0  # Target is always 0.0 in PID mode
            norm_target_y = 0.0
            norm_dt = (dt_ms / 33.0) - 1.0

            history_buffer.append(
                [norm_cnn_x, norm_cnn_y, norm_target_x, norm_target_y, norm_dt]
            )

            mlp_ms = 0.0
            if len(history_buffer) == 1:
                # Flatten the 1x5 buffer into a 1x5 input array
                mlp_input = (
                    np.array(history_buffer, dtype=np.float32).flatten().reshape(1, -1)
                )
                mlp_out = mlp_session.run(None, {mlp_input_name: mlp_input})[0][0]

                final_x = float(mlp_out[0])
                final_y = float(mlp_out[1])
                deriv_x, deriv_y = 0.0, 0.0

                mlp_ms = (time.perf_counter() - mlp_t0) * 1000.0
            else:
                # Buffer not full, fallback to raw CNN tracking
                final_x = centered_touch_x
                final_y = centered_touch_y
                deriv_x, deriv_y = 0.0, 0.0

            # ----------------------------------------------------------------
            # Prediction Gate
            # ----------------------------------------------------------------
            gated_x, gated_y, gate_reason = gate.filter(final_x, final_y)

            if gate_reason == "no_ball":
                # Ball not yet confirmed — audio can still be processed but
                # do NOT send anything to the firmware.
                command = audio_receiver.get_latest_command()
                if command:
                    print(f"\n[AUDIO] (gate=no_ball) Heard: {command} — waiting for ball\n")
                continue

            if gate_reason not in ("ok", "seeded"):
                print(
                    f"  ⚠ Gate [{gate_reason}] rejected ({final_x:+.1f}, {final_y:+.1f}) mm "
                    f"→ holding ({gated_x:+.1f}, {gated_y:+.1f}) mm"
                )
            final_x, final_y = gated_x, gated_y

            # ----------------------------------------------------------------
            # Process Audio Commands
            # ----------------------------------------------------------------
            command = audio_receiver.get_latest_command()
            if command:
                print(f"\n[AUDIO] Heard command: {command}\n")

            # Pass final_x, final_y into the state machine to use as the hold point
            state_machine.process_command(command, final_x, final_y)

            target_x, target_y = state_machine.get_target_coords()

            # ----------------------------------------------------------------
            # Serial Transmission
            # ----------------------------------------------------------------
            try:
                # Payload: cam_x, cam_y, target_x, target_y
                payload = f"{final_x:.2f},{final_y:.2f},{target_x:.2f},{target_y:.2f}\n".encode(
                    "ascii"
                )

                # TODO: When firmware is updated to support derivatives directly, swap to this:
                # payload = f"{final_x:.2f},{final_y:.2f},{deriv_x:.2f},{deriv_y:.2f}\n".encode('ascii')

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
                print(
                    f"[{phase}] Ball: X={final_x:+6.1f} Y={final_y:+6.1f} mm | "
                    f"Target: {state_machine.current_target_name} at X={target_x:.1f} Y={target_y:.1f} | "
                    f"Cmd: {state_machine._last_command} | FPS: {fps:.1f} | "
                    f"Total={total_ms:.1f}ms (ArUco={aruco_ms:.1f}ms, CNN={cnn_ms:.1f}ms, MLP={mlp_ms:.1f}ms, Audio={audio_ms:.1f}ms)"
                )
                last_status_t = end_t

            # Optional: Display for debugging
            if not args.headless:
                cv2.circle(
                    frame, (int(ball_frame_x), int(ball_frame_y)), 10, (0, 0, 255), -1
                )
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.imshow("ArUco + CNN + MLP Tracker (ONNX)", frame)
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
