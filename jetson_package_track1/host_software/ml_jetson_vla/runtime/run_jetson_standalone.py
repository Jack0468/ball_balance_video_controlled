"""Jetson entry point for the small-class expert pipeline (Track 1 of the Jetson port
plan, 2026-08-18). Runtime-target port of `host_software/main_onnx_shared_vision_audio.py`
-- same pipeline (ArUco homography+warp -> shared_vision_backbone_v2 CNN -> marker
detection -> audio ONNX -> state machine -> PredictionGate), same wire protocol. Reuses
that script's `PredictionGate`, `preprocess_warped`, `px_to_touch_mm`, `sigmoid`, and
manifest/margin constants directly (import, not copy-paste) since none of that logic is
platform-specific -- only what changes below is:

  - Camera open: unchanged code path (`src.receivers.USBReceiver`), but on Linux/Jetson
    its `cv2.CAP_DSHOW` attempt fails and it falls through to an explicit `cv2.CAP_V4L2`
    request. **Correction, confirmed on real hardware 2026-09-15**: this docstring
    originally assumed the untargeted default backend (no explicit flag) would land on
    V4L2 -- it doesn't, it lands on GStreamer, which silently ignores the
    CAP_PROP_FRAME_WIDTH/HEIGHT `.set()` calls and opens at the camera's native 1280x720
    instead of the requested 640x480, no error raised. `receivers.py` now requests V4L2
    explicitly (confirmed present via `cv2.videoio_registry.getBackends()`) rather than
    trusting the default. Lesson: "falls through to X" is a claim to verify, not assume,
    even when it sounds like standard OpenCV behavior.
  - Serial port default: Windows' "COM7" fallback replaced with a Linux tty path. Real
    detection still goes through `find_stm32_port()` (`src.utils`), which is already
    OS-agnostic (matches on `pyserial` port description, not a Windows-specific string).
  - ONNX providers: CPUExecutionProvider only, matching this session's CPU-first decision
    for the small-class pipeline (~70-91K/13.5K params -- already real-time on a laptop
    CPU, no GPU work justified here). See Track 3 for where GPU/TensorRT actually matters.

**Phase A only**: the control net still runs on the STM32 (`RLControl.cpp`), unchanged.
This script sends the tagged `V,<seq>,ball_x,ball_y,target_x,target_y` ASCII form over
`SerialCoords.cpp` -- matching the laptop's `touch_logger.send_frame()` convention exactly
(previously this sent the untagged legacy 4-field form; both are accepted by
`SerialCoords.cpp`, but tagging gives every frame a host-supplied sequence number, needed
to join outbound `T,...` telemetry echoes back to the frame that produced them). It does
NOT use `ml_jetson_vla/core/control_net.py` or `RemoteStepControl.cpp`. Migrating
control-net inference onto the Jetson is Phase B (see `firmware/stm32_jetson_remote_control/`
once that exists), deferred pending the firmware telemetry-back work proposed in
`../stm32_interface/`.

Logic is wrapped behind `core.policy_interface.Policy` (`JetsonExpertPolicy` below) so a
future arm (Track 3's medium class, or a large-VLA arm) can be swapped in without touching
the camera/serial/runtime-loop plumbing in `main()`.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Optional

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
from src.touch_logger import TouchTelemetryLogger
from host_software.ml_vision.core.keyboard_command_receiver import KeyboardCommandReceiver
from host_software.ml_vision.core.scripted_command_sequencer import (
    ScriptedCommandSequencer,
    STANDARD_EVAL_SCHEDULE,
    random_schedule,
)

from host_software.ml_vision.data_processing.auto_label_shared_vision import (
    build_paper_corners,
    estimate_homography_from_aruco,
    load_manifest_full,
    warp_to_platform,
)
from host_software.ml_vision.core.marker_classifier import MarkerClassifier
from host_software.ml_vision.core.kalman_filter import KalmanFilter2D

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
from ml_jetson_vla.core.control_net import ControlNet
from ml_jetson_vla.runtime.motor_geometry import steps_to_angle
from ml_jetson_vla.runtime.session_recorder import SessionRecorder, SessionTouchTap

# --- Configuration ---
# Laptop default was "COM7". No physical default tty is more "correct" than another on
# Linux -- this is just a fallback for when find_stm32_port() (VID/PID + description
# matching, already OS-agnostic) comes up empty, not something to rely on normally.
SERIAL_PORT = "/dev/ttyACM0"
SERIAL_BAUD = 2000000


class TelemetryReader:
    """Non-blocking line reader for the STM32's uplink telemetry. Only used in
    --remote-control mode (Phase B): the firmware in
    firmware/stm32_jetson_remote_control/BallBalancingBot/TouchProbe.cpp emits
    `T,<seq>,<ms>,<touch_x>,<touch_y>,<valid>,<a>,<b>,<c>\\n` at 25Hz -- `a,b,c`
    are the REAL motor step positions ControlNet.infer() needs as `actual_steps`
    (trained on lagged motor state, not the commanded target -- see
    control_net.py's own docstring). `ser` must be opened non-blocking
    (timeout=0); this accumulates partial lines across calls the same way the
    firmware-side line buffers do."""

    def __init__(self) -> None:
        self._buf = b""
        self.last_steps: tuple = (0, 0, 0)  # matches home_motors()'s zero position at boot
        self.have_telemetry = False

    def poll(self, ser: "serial.Serial") -> None:
        if ser is None or ser.in_waiting <= 0:
            return
        self._buf += ser.read(ser.in_waiting)
        while b"\n" in self._buf:
            line, self._buf = self._buf.split(b"\n", 1)
            self._handle_line(line)

    def _handle_line(self, line: bytes) -> None:
        if not line.startswith(b"T,"):
            return  # '#' status lines and anything else -- not our format
        fields = line.decode("ascii", errors="ignore").split(",")
        if len(fields) != 9:
            return  # malformed -- drop rather than guess
        try:
            a, b, c = int(fields[6]), int(fields[7]), int(fields[8])
        except ValueError:
            return
        self.last_steps = (a, b, c)
        self.have_telemetry = True


class JetsonExpertPolicy(Policy):
    """Small-class expert pipeline (vision CNN + marker detection + audio + state
    machine), Phase A: computes a target in mm, does NOT compute step targets -- the
    STM32's own `RLControl.cpp` still does that. Owns the CNN/audio ONNX sessions, the
    marker classifier, the state machine, and the prediction gate; `act()` is one frame."""

    def __init__(self, script_dir: str, marker_gate_mm: float, jump_gate_mm: float,
                 gate_ema_alpha: float, seed_window: int, seed_consistency_mm: float,
                 lost_frames: int, mask_threshold: float,
                 dummy_audio: bool = False, mic_device: Optional[str] = None,
                 kalman: Optional[KalmanFilter2D] = None,
                 eval_sequence: bool = False,
                 random_sequence: bool = False,
                 random_sequence_duration: float = 200.0,
                 random_sequence_min_interval: float = 3.0,
                 random_sequence_max_interval: float = 20.0,
                 random_sequence_seed: Optional[int] = None) -> None:
        if eval_sequence and random_sequence:
            raise ValueError(
                "--eval-sequence and --random-sequence are mutually exclusive -- both "
                "are scripted command sources, pick one."
            )
        cnn_path = os.path.abspath(
            os.path.join(script_dir, "ml_vision/models/shared_vision_backbone_v2/shared_vision_backbone_best.onnx")
        )
        if not os.path.exists(cnn_path):
            raise FileNotFoundError(f"ONNX vision model not found at {cnn_path}")

        # Matches main_onnx_shared_vision_audio.py's --dummy-audio/--scripted-sequence
        # precedence pattern (same KeyboardCommandReceiver/ScriptedCommandSequencer
        # swaps) -- the audio ONNX file is only required to exist when it's actually
        # going to be loaded, i.e. neither of the no-audio-model modes is active.
        if not dummy_audio and not eval_sequence and not random_sequence:
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

        if eval_sequence:
            # docs/EVALUATION_STRATEGY.md's Standardized Evaluation Sequence -- takes
            # precedence over --dummy-audio (matching the laptop's own
            # scripted-sequence-first precedence), since a scripted comparison run and
            # manual keyboard testing are mutually exclusive use cases.
            self.audio_receiver = ScriptedCommandSequencer(schedule=STANDARD_EVAL_SCHEDULE)
        elif random_sequence:
            # Track 4 unattended data-collection smoke test -- same precedence tier as
            # eval_sequence (both are scripted sources, ahead of --dummy-audio), mutually
            # exclusive with eval_sequence (checked above).
            self.audio_receiver = ScriptedCommandSequencer(
                schedule=random_schedule(
                    total_duration_s=random_sequence_duration,
                    min_interval_s=random_sequence_min_interval,
                    max_interval_s=random_sequence_max_interval,
                    seed=random_sequence_seed,
                )
            )
        elif dummy_audio:
            self.audio_receiver = KeyboardCommandReceiver()
        else:
            self.audio_receiver = AudioCommandReceiverONNX(audio_path, mic_device=mic_device)
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
            kalman=kalman,
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
            kalman=self.gate.kalman,
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
        self.last_debug["raw_ball_xy_mm"] = (raw_x, raw_y)
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
    parser.add_argument("--kalman", action="store_true", help="Replace PredictionGate's EMA smoothing with a constant-velocity Kalman filter (jump-gate/seed/no-ball logic unchanged). Needs R/Q from --kalman-params to be trustworthy -- see kalman_filter.py")
    parser.add_argument("--kalman-params", type=str, default=None, help="Path to a JSON file with 'r' (2x2) and 'q' (4x4) matrices, as produced by estimate_kalman_noise_params.py --output-json. Without this, --kalman uses illustrative fallback values (not validated -- see kalman_filter.py docstring)")
    parser.add_argument("--log-csv", type=str, default="auto", help="Ground-truth telemetry CSV via TouchTelemetryLogger/TouchProbe.cpp's existing 'T,...' uplink (confirmed live on this firmware 2026-09-15, no firmware change needed): 'auto' (timestamped file in data/01_bronze/evaluation/), 'off' (disable persistence -- the serial I/O thread still runs), or an explicit path. Only active when NOT --remote-control (Phase B's TelemetryReader already owns reading this same serial handle -- see module docstring).")
    parser.add_argument("--quiet-mcu", action="store_true", help="Suppress TouchTelemetryLogger's per-line MCU status printing")
    parser.add_argument(
        "--record-track4-session", action="store_true",
        help="Also write a session-structured Track 4 bronze capture "
             "(data/01_bronze/session_jetson_track4_<timestamp>/{telemetry.csv,rgb_video.mp4}), "
             "picked up by ml_multimodal/data_processing/generate_vla_dataset.py's "
             "existing session_* glob and by this directory's own "
             "data_processing/convert_to_lerobot.py. Requires a live serial connection "
             "and Phase A (--remote-control not set), since it taps the same 'T,...' "
             "uplink TouchTelemetryLogger already reads (see session_recorder.py's "
             "SessionTouchTap) rather than opening a second serial reader. Independent "
             "of --log-csv, which still controls the separate evaluation ground-truth CSV.",
    )
    parser.add_argument(
        "--dummy-audio", action="store_true",
        help="Use typed keyboard commands instead of the trained audio model -- for "
             "testing state-machine/target-switching without a working mic (temporary "
             "testing aid, see keyboard_command_receiver.py). Same flag name/behavior as "
             "main_onnx_shared_vision_audio.py's --dummy-audio -- use this for Jetson "
             "bring-up before the mic/audio path is wired up there; audio (possibly the "
             "NeMo-finetuned model, still exploratory as of 2026-09-15 -- see "
             "ml_audio/docs/plans/audio_eval_notebook_refactor_plan.md) gets added back "
             "as a separate step, not blocking this.",
    )
    parser.add_argument(
        "--eval-sequence", action="store_true",
        help="Run docs/EVALUATION_STRATEGY.md's Standardized Evaluation Sequence "
             "(ScriptedCommandSequencer + STANDARD_EVAL_SCHEDULE) instead of live audio "
             "or keyboard input -- the reproducible, timed 10-command comparison protocol "
             "for the PID/Expert-Vision-RL/VLA arms. Takes precedence over --dummy-audio. "
             "Combine with --log-csv (on by default) to actually collect the comparison "
             "data; --kalman/--kalman-params recommended for consistency with other runs.",
    )
    parser.add_argument(
        "--random-sequence", action="store_true",
        help="Run a randomized command schedule (ScriptedCommandSequencer + "
             "random_schedule()) instead of live audio or keyboard input -- for "
             "unattended Track 4 data-collection smoke tests: randomly-ordered "
             "commands (no immediate repeats) over a fixed total session duration, "
             "each held a randomized gap (see --random-sequence-min-interval/"
             "--random-sequence-max-interval), drawn from the confirmed active "
             "vocabulary. Command count is a consequence of the random gaps, not a "
             "fixed input. Same precedence tier as --eval-sequence (both are "
             "scripted sources, ahead of --dummy-audio); mutually exclusive with "
             "--eval-sequence. Combine with --record-track4-session to actually "
             "capture the run.",
    )
    parser.add_argument(
        "--random-sequence-duration", type=float, default=200.0,
        help="Total session duration in seconds for the --random-sequence schedule "
             "(default: 200.0). The last command fires somewhere before this, never "
             "at or past it.",
    )
    parser.add_argument(
        "--random-sequence-min-interval", type=float, default=3.0,
        help="Minimum randomized gap in seconds between consecutive commands in the "
             "--random-sequence schedule (default: 3.0).",
    )
    parser.add_argument(
        "--random-sequence-max-interval", type=float, default=20.0,
        help="Maximum randomized gap in seconds between consecutive commands in the "
             "--random-sequence schedule (default: 20.0, exclusive upper bound).",
    )
    parser.add_argument(
        "--random-sequence-seed", type=int, default=None,
        help="Optional seed for the --random-sequence schedule, to reproduce a specific "
             "problematic session later. Uses a local random.Random instance -- global "
             "random state is untouched. Default: unseeded (a different sequence each run).",
    )
    parser.add_argument(
        "--mic-device", type=str, default="JBCW036",
        help="Substring to match the mic's input device name (default: JBCW036, the "
             "USB camera's built-in mic). Confirmed 2026-09-15: sounddevice has no "
             "reliable cross-machine default here -- Linux/the Jetson enumerates audio "
             "devices completely differently than Windows, so pin by name rather than "
             "index or trusting the OS default. Use "
             "deployment/test_microphone.py --list to see device names on this machine.",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--remote-control", action="store_true",
        help="Phase B: run control_net.py on the Jetson and send precomputed step targets "
             "('S,<a>,<b>,<c>') instead of raw coordinates ('V,<seq>,...'). Requires the "
             "firmware in firmware/stm32_jetson_remote_control/BallBalancingBot/, NOT the "
             "default RL-on-device firmware -- the two speak incompatible wire protocols. "
             "Default (unset) is unchanged Phase A behaviour.",
    )
    args = parser.parse_args()

    if args.port == "auto":
        detected_port = find_stm32_port()
        if detected_port:
            print(f"Auto-detected STM32 on {detected_port}")
            args.port = detected_port
        else:
            args.port = SERIAL_PORT
            print(f"Could not auto-detect STM32. Defaulting to {args.port}")

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

    policy = JetsonExpertPolicy(
        script_dir=_HOST_SOFTWARE_DIR,
        marker_gate_mm=args.marker_gate_mm,
        jump_gate_mm=args.jump_gate_mm,
        gate_ema_alpha=args.gate_ema_alpha,
        seed_window=args.seed_window,
        seed_consistency_mm=args.seed_consistency_mm,
        lost_frames=args.lost_frames,
        mask_threshold=args.mask_threshold,
        dummy_audio=args.dummy_audio,
        mic_device=args.mic_device,
        kalman=kalman,
        eval_sequence=args.eval_sequence,
        random_sequence=args.random_sequence,
        random_sequence_duration=args.random_sequence_duration,
        random_sequence_min_interval=args.random_sequence_min_interval,
        random_sequence_max_interval=args.random_sequence_max_interval,
        random_sequence_seed=args.random_sequence_seed,
    )

    try:
        ser = serial.Serial(args.port, SERIAL_BAUD, timeout=0)
        print(f"Connected to STM32 on {args.port} at {SERIAL_BAUD} baud.")
    except Exception:
        print(f"Could not open serial port {args.port}. Continuing in dry-run mode.")
        ser = None

    # Phase A only -- Phase B's TelemetryReader below already owns reading this same
    # serial handle for a DIFFERENT, 8-field wire format (firmware/stm32_jetson_remote_control/),
    # so the two must never both read at once. TouchProbe.cpp's 11-field "T,..." uplink
    # (matching touch_logger.py's parser) is already being sent by the CURRENT firmware
    # (firmware/stm32_ml_control_and_vision/) regardless of whether anything reads it --
    # confirmed live on real hardware 2026-09-15 -- this was simply never wired up on the
    # Jetson side before now. No firmware change needed.
    touch_logger = None
    session_recorder = None
    if ser is not None and not args.remote_control:
        csv_path = None
        if args.log_csv != "off":
            if args.log_csv == "auto":
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_path = os.path.join(_HOST_SOFTWARE_DIR, "data", "01_bronze", "evaluation", f"ground_truth_jetson_{stamp}.csv")
            else:
                csv_path = args.log_csv
        # SessionTouchTap is a drop-in TouchTelemetryLogger subclass (same constructor,
        # same CSV/thread behavior) that additionally caches the latest touch/motor
        # reading for --record-track4-session -- see session_recorder.py's docstring for
        # why this is a subclass on the SAME worker thread rather than a second reader.
        logger_cls = SessionTouchTap if args.record_track4_session else TouchTelemetryLogger
        touch_logger = logger_cls(ser, csv_path, print_status_lines=not args.quiet_mcu)
        touch_logger.start()
        if args.record_track4_session:
            bronze_root = os.path.join(_HOST_SOFTWARE_DIR, "data", "01_bronze")
            session_recorder = SessionRecorder(bronze_root, fps=30.0)
    elif args.record_track4_session:
        print(
            "[track4-session] --record-track4-session requires a live serial connection "
            "and Phase A (not --remote-control) -- session recording disabled for this run."
        )

    control_net = None
    telemetry = None
    if args.remote_control:
        control_net = ControlNet()
        telemetry = TelemetryReader()
        print(f"Remote control ON -- ControlNet loaded ({control_net.total_params} params). "
              f"Sending 'S,<a>,<b>,<c>' step targets, expects firmware/stm32_jetson_remote_control/.")

    if args.udp:
        receiver = UDPReceiver(port=args.udp_port, width=640, height=480)
    else:
        # fps=30: this camera's real capability, confirmed via v4l2-ctl (2026-09-09) --
        # MJPG at 640x480 only has a single discrete interval, 30fps, not the
        # USBReceiver default of 60. See docs/CAMERA_HARDWARE.md for the full finding.
        receiver = USBReceiver(camera_id=args.cam_id, fps=30)

    print("Waiting for camera feed...")
    frame = None
    while frame is None:
        frame = receiver.get_latest_frame()
        time.sleep(0.1)

    mode_label = "Phase B -- control_net.py runs the policy here" if args.remote_control else "Phase A -- STM32 keeps its own control net"
    print(f"Starting Jetson standalone loop (Track 1, {mode_label})... Ctrl+C to quit")
    last_status_t = 0.0
    last_frame_t = None  # for --remote-control's actual_dt; None until the first successful frame
    seq = 0
    last_audio_command: Optional[str] = None  # persists across frames like target_x/y already does

    try:
        while True:
            frame = receiver.get_latest_frame()
            if frame is None:
                continue

            start_t = time.perf_counter()
            if telemetry is not None:
                telemetry.poll(ser)  # drain any T,... lines that arrived since the last frame

            command = policy.audio_receiver.get_latest_command()
            if command:
                last_audio_command = command
            cmd_out = policy.act(frame, command, state={})

            if (
                policy.last_debug.get("gate_reason") == "seeded"
                and isinstance(policy.audio_receiver, ScriptedCommandSequencer)
            ):
                # PredictionGate's AWAITING_BALL -> TRACKING transition (see
                # main_onnx_shared_vision_audio.py's PredictionGate._handle_awaiting) --
                # fires exactly once per run, the first frame the ball is confirmed. This
                # is what --eval-sequence/--scripted-sequence should treat as "the
                # platform is ready," not construction time, so the schedule's clock
                # doesn't burn real time on camera warm-up / no-ball waiting. Safe to
                # call every frame after that too -- begin() is idempotent.
                policy.audio_receiver.begin()

            if cmd_out is None:
                if command:
                    print(f"\n[AUDIO] (no ball) Heard: {command}\n")
                if touch_logger is not None:
                    touch_logger.send_raw(b"L\n")
                continue

            try:
                bx, by = policy.last_debug["ball_xy_mm"]

                if session_recorder is not None:
                    # SessionTouchTap.get_latest_touch() is a best-effort, non-blocking
                    # snapshot -- the 'T,...' uplink runs at its own ~25Hz cadence,
                    # independent of this loop's rate, so it may lag or (early in a run)
                    # be None. theta_a/b/c come from the same motor_geometry.py port
                    # used nowhere in the control path -- see its docstring for why no
                    # origin-offset subtraction is needed here.
                    touch_snapshot = (
                        touch_logger.get_latest_touch()
                        if isinstance(touch_logger, SessionTouchTap)
                        else None
                    )
                    if touch_snapshot is not None:
                        t_x, t_y = touch_snapshot["touch_x"], touch_snapshot["touch_y"]
                        theta_a = steps_to_angle(touch_snapshot["motor_a"])
                        theta_b = steps_to_angle(touch_snapshot["motor_b"])
                        theta_c = steps_to_angle(touch_snapshot["motor_c"])
                    else:
                        t_x = t_y = theta_a = theta_b = theta_c = None
                    session_recorder.log(
                        frame,
                        int(time.time() * 1000),
                        cmd_out.target_x_mm,
                        cmd_out.target_y_mm,
                        t_x, t_y,
                        theta_a, theta_b, theta_c,
                        last_audio_command,
                    )

                if args.remote_control:
                    # actual_dt: real measured time since the last control cycle, not a
                    # fixed constant -- this is the whole point of Phase B (see
                    # control_net.py's CONTROL_DT comment: RLControl.cpp's own "30Hz"
                    # label was never actually enforced either, but here it's explicit).
                    now_t = time.perf_counter()
                    actual_dt = (now_t - last_frame_t) if last_frame_t is not None else (1.0 / 24.0)
                    last_frame_t = now_t
                    steps = control_net.infer(
                        bx, by, cmd_out.target_x_mm, cmd_out.target_y_mm,
                        telemetry.last_steps, actual_dt,
                    )
                    payload = f"S,{steps[0]},{steps[1]},{steps[2]}\n".encode("ascii")
                    if ser is not None:
                        ser.write(payload)
                elif touch_logger is not None:
                    # touch_logger's worker thread is the sole owner of the serial handle
                    # once constructed -- never call ser.write() directly alongside it (see
                    # PROJECT_LOGBOOK.md 19/08 for the throughput-collapse this caused on
                    # the laptop the one time two threads touched the handle concurrently).
                    seq += 1
                    raw_x, raw_y = policy.last_debug["raw_ball_xy_mm"]
                    touch_logger.send_frame(seq, bx, by, cmd_out.target_x_mm, cmd_out.target_y_mm, raw_x=raw_x, raw_y=raw_y)
                else:
                    # Tagged form (matches touch_logger.send_frame() on the laptop) --
                    # see module docstring. seq wraps at 2**32 to match SerialCoords.cpp's
                    # uint32 last_seq field; the firmware only uses it for ordering/echo,
                    # not as a global counter, so wraparound is harmless.
                    payload = f"V,{seq},{bx:.2f},{by:.2f},{cmd_out.target_x_mm:.2f},{cmd_out.target_y_mm:.2f}\n".encode("ascii")
                    seq = (seq + 1) % (2**32)
                    if ser is not None:
                        ser.write(payload)
            except Exception as e:
                print(f"Serial Error: {e}")

            total_ms = (time.perf_counter() - start_t) * 1000.0
            now = time.perf_counter()
            if args.verbose or (now - last_status_t) >= 1.0:
                markers_str = ", ".join(f"{c}=({x:+.0f},{y:+.0f})" for c, (x, y) in policy.last_debug["markers"].items()) or "none"
                telemetry_str = ""
                if telemetry is not None:
                    tag = "steps" if telemetry.have_telemetry else "steps(no telemetry yet, using 0,0,0)"
                    telemetry_str = f" | {tag}={telemetry.last_steps}"
                print(
                    f"[{policy.last_debug['gate_reason']}] Ball: X={bx:+6.1f} Y={by:+6.1f} mm | "
                    f"Target: X={cmd_out.target_x_mm:.1f} Y={cmd_out.target_y_mm:.1f} | "
                    f"Markers: {markers_str} | Frame={total_ms:.1f}ms{telemetry_str}"
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
        if touch_logger is not None:
            touch_logger.stop()
        if session_recorder is not None:
            session_recorder.stop()
        if ser:
            ser.close()
        if not args.headless:
            # Confirmed 2026-09-15: some Jetson OpenCV builds (this one drifted from the
            # earlier-verified apt python3-opencv 4.5.4 to a pip-style 4.12.0 with no GTK
            # backend at all) raise "function is not implemented" even on
            # destroyAllWindows() -- not just imshow(). Since --headless never calls
            # imshow() in the first place, there are no windows to destroy either; gate
            # this the same way so a correctly-headless run can't crash in cleanup on a
            # GUI-less build. Non-headless use on such a build still fails at imshow()
            # itself, which is the right place for that failure, not here.
            cv2.destroyAllWindows()
        print("Jetson standalone loop stopped.")


if __name__ == "__main__":
    main()
