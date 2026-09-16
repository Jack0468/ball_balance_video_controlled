"""Live laptop inference entry point for shared_vision_backbone_v2, using the
NeMo pretrained-backbone audio classifier instead of the custom-CNN ONNX one.

Forked from main_onnx_shared_vision_audio.py rather than adding a flag to it,
per explicit instruction (a separate new entry point, not a change to the
one other sessions currently have in flight). Everything vision-side
(PredictionGate, warp_to_platform preprocessing, marker detection, the main
loop's stage structure) is IMPORTED unchanged from that file, not
re-implemented -- the only real difference is STAGE 6's audio backend:
NemoAudioCommandReceiver (ml_audio/evaluations/nemo_live_receiver.py) in
place of AudioCommandReceiverONNX.

This is "Option B" from docs/plans/audio_eval_notebook_refactor_plan.md's
"Production Integration Check" section, not Option A -- it runs the full
NeMo/PyTorch model directly (model.forward(), MFCC preprocessing included),
NOT the NeMo ONNX export. That's a deliberate, informed choice already
documented there (Option A needs a from-scratch ONNX-compatible MFCC
reimplementation that doesn't exist yet), not an oversight. Consequence:
this pipeline's audio path now depends on nemo_toolkit[asr] + torch, not
just onnxruntime + sounddevice -- confirm that installs cleanly in
ball_balance_env (this project's standard interpreter, per CLAUDE.md) before
relying on this for anything beyond local testing; it has so far only been
verified in a separate nemo_local conda env.

IMPORTANT, read before trusting this for anything but integration testing:
- The default checkpoint below is a PRELIMINARY, half-trained NOISE_MIX
  checkpoint (see plan doc "First NOISE_MIX Data Point (2026-09-15)") --
  wired up at the user's explicit request to test this entry point's
  plumbing, not because it's been validated as the best available
  checkpoint. Neither of its two seeds has shown a live-stream `backward`
  win yet, and neither run finished training. The plan's own currently-
  recommended checkpoint is models/nemo_matchboxnet_v1/matchboxnet_finetuned.nemo
  (9/11 live-stream, the best on record) -- pass --audio-model to use that
  instead once you actually want a real accuracy number, not a plumbing
  check.
- Verified so far: this file imports cleanly, --dummy-audio and
  --scripted-sequence paths exercise the same vision/state-machine code the
  existing entry point already uses, and NemoAudioCommandReceiver's file-
  stream mode (used by the eval scripts) still passes its own regression
  check after the live-mic support added here. NOT yet verified: an actual
  live microphone, or the real robot end-to-end -- per this session's scope
  (audio/vision development, not physical hardware), that check needs to
  happen on your machine, not here.
"""

import argparse
import collections
import json
import os
import sys
import time
from datetime import datetime

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
from src.touch_logger import TouchTelemetryLogger

from host_software.ml_vision.data_processing.auto_label_shared_vision import (
    build_paper_corners,
    estimate_homography_from_aruco,
    load_manifest_full,
    warp_to_platform,
)
from host_software.ml_vision.core.marker_classifier import MarkerClassifier
from host_software.ml_vision.core.keyboard_command_receiver import KeyboardCommandReceiver
from host_software.ml_vision.core.scripted_command_sequencer import ScriptedCommandSequencer
from host_software.ml_vision.core.kalman_filter import KalmanFilter2D

# Reused, not reimplemented -- see module docstring.
from host_software.main_onnx_shared_vision_audio import (
    GROUND_TRUTH_MANIFEST,
    INPUT_SIZE,
    PAPER_MARGIN_MM,
    PredictionGate,
    _ARUCO_MARKER_CENTRES_MM_RAW,
    preprocess_warped,
    px_to_touch_mm,
    sigmoid,
)

from host_software.ml_audio.evaluations.nemo_live_receiver import NemoAudioCommandReceiver

# --- Configuration ---
SERIAL_PORT = "COM7"
SERIAL_BAUD = 2000000

# Preliminary/half-trained -- see module docstring. Override with --audio-model
# to use the plan's actually-recommended checkpoint
# (models/nemo_matchboxnet_v1/matchboxnet_finetuned.nemo, 9/11 live-stream).
DEFAULT_NEMO_MODEL = os.path.join(
    "ml_audio", "models", "nemo_matchboxnet_3x1x64_noisemix_seed0",
    "matchboxnet_3x1x64_noisemix_finetuned_seed0.nemo",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="shared_vision_backbone_v2 -> CNN + Marker Detection + NeMo Audio Tracker")
    parser.add_argument("--cam_id", type=int, default=1, help="Camera ID for USB mode")
    parser.add_argument("--cam-width", type=int, default=640, help="Requested camera capture width -- use probe_camera_modes.py to find what this camera actually delivers before assuming a value here")
    parser.add_argument("--cam-height", type=int, default=480, help="Requested camera capture height")
    parser.add_argument("--cam-auto-exposure", type=float, default=None, help="Value for cv2.CAP_PROP_AUTO_EXPOSURE (backend-dependent convention -- see probe_camera_modes.py). Default: leave the camera's power-on setting untouched")
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
    parser.add_argument("--settle-radius-mm", type=float, default=2.0, help="Dead-band (mm): transmitted position only moves once the smoothed estimate drifts past this from what's currently being sent (0 to disable)")
    parser.add_argument("--kalman", action="store_true", help="Replace PredictionGate's EMA smoothing with a constant-velocity Kalman filter (jump-gate/seed/no-ball logic unchanged). Needs R/Q from --kalman-params to be trustworthy -- see kalman_filter.py")
    parser.add_argument("--kalman-params", type=str, default=None, help="Path to a JSON file with 'r' (2x2) and 'q' (4x4) matrices, as produced by estimate_kalman_noise_params.py --output-json. Without this, --kalman uses illustrative fallback values (not validated -- see kalman_filter.py docstring)")
    parser.add_argument("--mask-threshold", type=float, default=0.5, help="Sigmoid threshold for marker segmentation mask")
    parser.add_argument("--mlp", action="store_true", help="Apply the shared-vision MLP time corrector on top of raw CNN output (default off -- see PROJECT_LOGBOOK for validation results)")
    parser.add_argument("--mlp-model-path", type=str, default=None, help="Override path to the MLP corrector .pth (default: models/mlp_corrector_shared_vision_v1/mlp_corrector_best.pth)")
    parser.add_argument("--audio-model", type=str, default=None, help=f"Path to a .nemo checkpoint (default: preliminary NOISE_MIX checkpoint at {DEFAULT_NEMO_MODEL} -- see module docstring; pass the plan's recommended ml_audio/models/nemo_matchboxnet_v1/matchboxnet_finetuned.nemo for a validated number instead)")
    parser.add_argument("--mic-device", type=str, default=None, help="Input device name substring or index (default: OS default input)")
    parser.add_argument("--dummy-audio", action="store_true", help="Use typed keyboard commands instead of the trained audio model -- for testing state-machine/target-switching without a working mic (temporary testing aid, see keyboard_command_receiver.py)")
    parser.add_argument("--scripted-sequence", action="store_true", help="Drive the target with a fixed, reproducible command schedule instead of live audio/keyboard input -- for collecting controlled telemetry (settle window + swept motion) to design Kalman filter noise parameters. See scripted_command_sequencer.py")
    parser.add_argument("--log-csv", type=str, default="auto", help="Ground-truth telemetry CSV: 'auto' (timestamped file in data/01_bronze/evaluation/), 'off' (disable persistence -- the serial I/O thread still runs), or an explicit path")
    parser.add_argument("--quiet-mcu", action="store_true", help="Suppress TouchTelemetryLogger's per-line MCU status printing")
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

    # ---- 1. Model Init (ONNX vision + NeMo audio) ----
    cnn_path = os.path.abspath(os.path.join(script_dir, "ml_vision/models/shared_vision_backbone_v2/shared_vision_backbone_best.onnx"))
    if not os.path.exists(cnn_path):
        print(f"Error: ONNX model not found at {cnn_path}")
        return

    audio_path = os.path.abspath(os.path.join(script_dir, args.audio_model or DEFAULT_NEMO_MODEL))
    if not args.dummy_audio and not args.scripted_sequence and not os.path.exists(audio_path):
        print(f"Error: NeMo audio model not found at {audio_path}")
        return

    print("Loading ONNX vision session...")
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

    mic_device = args.mic_device
    if mic_device is not None and mic_device.isdigit():
        mic_device = int(mic_device)

    if args.scripted_sequence:
        audio_receiver = ScriptedCommandSequencer()
    elif args.dummy_audio:
        audio_receiver = KeyboardCommandReceiver()
    else:
        print("Initializing NeMo Audio Receiver...")
        audio_receiver = NemoAudioCommandReceiver(audio_path, mic_device=mic_device)
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
        ser = serial.Serial(args.port, SERIAL_BAUD, timeout=0.02)
        print(f"Connected to STM32 on {args.port} at {SERIAL_BAUD} baud.")
    except Exception:
        print(f"Could not open serial port {args.port}. Continuing in dry-run mode.")
        ser = None

    touch_logger = None
    if ser is not None:
        csv_path = None
        if args.log_csv != "off":
            if args.log_csv == "auto":
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_path = os.path.join(script_dir, "data", "01_bronze", "evaluation", f"ground_truth_{stamp}.csv")
            else:
                csv_path = args.log_csv
        touch_logger = TouchTelemetryLogger(ser, csv_path, print_status_lines=not args.quiet_mcu)
        touch_logger.start()

    # ---- 4. Camera/Receiver Init ----
    if args.udp:
        receiver = UDPReceiver(port=args.udp_port, width=640, height=480)
    else:
        receiver = USBReceiver(
            camera_id=args.cam_id,
            width=args.cam_width,
            height=args.cam_height,
            auto_exposure=args.cam_auto_exposure,
        )

    print("Waiting for camera feed...")
    frame = None
    while frame is None:
        frame = receiver.get_latest_frame()
        time.sleep(0.1)

    print("Starting shared_vision_backbone_v2 -> Marker Detection + NeMo Audio Tracker loop... (press Ctrl+C to quit)")

    kalman = None
    if args.kalman:
        r, q = None, None
        if args.kalman_params:
            with open(args.kalman_params) as f:
                params = json.load(f)
            r = np.array(params["r"], dtype=np.float64)
            q = np.array(params["q"], dtype=np.float64)
            print(f"Kalman filter: loaded R/Q from {args.kalman_params}")
        else:
            print("Kalman filter: --kalman-params not given -- using illustrative fallback R/Q (NOT validated, see kalman_filter.py)")
        kalman = KalmanFilter2D(r=r, q=q)

    gate = PredictionGate(
        marker_centres=_ARUCO_CENTRES_CENTRED,
        marker_radius_mm=args.marker_gate_mm,
        jump_threshold_mm=args.jump_gate_mm,
        ema_alpha=args.gate_ema_alpha,
        seed_window=args.seed_window,
        seed_consistency_mm=args.seed_consistency_mm,
        lost_frames_threshold=args.lost_frames,
        settle_radius_mm=args.settle_radius_mm,
        kalman=kalman,
    )
    print(
        f"PredictionGate: marker_gate={args.marker_gate_mm:.0f}mm, jump_gate={args.jump_gate_mm:.0f}mm, "
        f"seed_window={args.seed_window} frames @ {args.seed_consistency_mm:.0f}mm, lost_threshold={args.lost_frames} frames, "
        f"settle_radius={args.settle_radius_mm:.1f}mm, smoothing={'kalman' if kalman is not None else 'ema'}"
    )
    print("  Waiting for ball to be placed on platform...")

    last_status_t: float = 0.0
    last_frame_time = time.perf_counter()
    seq = 0

    try:
        while True:
            frame = receiver.get_latest_frame()
            if frame is None:
                continue

            start_t = time.perf_counter()
            dt_ms = (start_t - last_frame_time) * 1000.0
            last_frame_time = start_t
            if dt_ms > 100.0:
                print(f"  ⚠ Stall detected ({dt_ms:.0f}ms since last frame) -- resetting gate smoothing")
                dt_ms = 33.0
                gate.reset_smoothing()
                if mlp_window is not None:
                    mlp_window.clear()
                if touch_logger is not None:
                    touch_logger.send_raw(b"L\n")

            # --- STAGE 1: ArUco Homography + Perspective Warp ---
            aruco_t0 = time.perf_counter()
            aruco_homography = estimate_homography_from_aruco(frame, aruco_lookup)
            aruco_ms = (time.perf_counter() - aruco_t0) * 1000.0

            if aruco_homography is None:
                print(f"Gate closed - Insufficient ArUco markers ({aruco_ms:.1f}ms)")
                if touch_logger is not None:
                    touch_logger.send_raw(b"L\n")
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
            was_tracking = gate.ball_on_platform
            gated_x, gated_y, gate_reason = gate.filter(final_x, final_y, dt_ms)

            if gate_reason == "seeded" and isinstance(audio_receiver, ScriptedCommandSequencer):
                audio_receiver.begin()

            if gate_reason == "no_ball":
                if mlp_window is not None:
                    mlp_window.clear()
                if was_tracking and not gate.ball_on_platform:
                    state_machine.on_ball_lost()
                command = audio_receiver.get_latest_command()
                if command:
                    print(f"\n[AUDIO] (gate=no_ball) Heard: {command} -- waiting for ball\n")
                if touch_logger is not None:
                    touch_logger.send_raw(b"L\n")
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
            state_machine.maybe_resume_previous_target(final_x, final_y)
            state_machine.maybe_auto_hold(final_x, final_y, marker_coords_xy)
            target_x, target_y = state_machine.get_target_coords()

            # --- STAGE 7: Serial Transmission ---
            seq += 1
            if touch_logger is not None:
                touch_logger.send_frame(seq, final_x, final_y, target_x, target_y, raw_x=raw_x, raw_y=raw_y)

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
                cv2.imshow("Shared Vision Backbone Tracker (NeMo Audio)", disp)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        pass
    finally:
        receiver.stop()
        audio_receiver.stop()
        if touch_logger is not None:
            touch_logger.stop()
        if ser:
            ser.close()
        cv2.destroyAllWindows()
        print("Inference loop stopped.")


if __name__ == "__main__":
    main()
