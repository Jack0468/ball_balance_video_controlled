import time
import queue
import torch
import numpy as np
import sounddevice as sd


from models.vision_model_residual import (
    load_yolo_model,
    load_mlp_corrector_iphone_v1_model,
    process_vision_frame,
)
from tools.coordinate_math import HomographyProjector
from tools.state_machine import TargetStateMachine
from tools.serial_control import SerialController

from tools.receivers import USBReceiver
from models.audio_model import load_audio_model

from tools.utils import find_stm32_port

# --- Configuration ---
TARGET_HZ = 30
LOOP_TIME = 1.0 / TARGET_HZ
AUDIO_SAMPLE_RATE = 16000
AUDIO_WINDOW_SAMPLES = int(AUDIO_SAMPLE_RATE * 1.25)  # 20,000 samples
AUDIO_CONFIDENCE_THRESHOLD = 0.60
AUDIO_DEBOUNCE_FRAMES = 3  # Require command to be held for N *audio* frames
AUDIO_EVERY = 3  # NEW: run audio inference every Nth loop (~10 Hz), not every loop
BALL_LOST_HOME_SEC = 3.0  # NEW: seconds of no-ball before homing/leveling the plate

SERIAL_PORT = "COM3"  # Update as needed
SERIAL_BAUD = 2000000


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting End-to-End System on {device} ---")

    # 1. Load All Models into memory
    yolo_model = load_yolo_model("models/yolo_markers_v3.pt", device)
    mlp_corrector = load_mlp_corrector_iphone_v1_model("models/mlp_corrector.pt", device)
    audio_model = load_audio_model("models/audio_model_weights.pt").to(device)

    # Auto-detect STM32 Port
    detected_port = find_stm32_port()
    if detected_port is None:
        print("❌ ERROR: Could not find STM32. Is it plugged in and turned on?")
        return
    else:
        print(f"✅ Found STM32 on port: {detected_port}")

    # Initialize hardware bridge and state machine using the detected port
    control = SerialController(
        port=detected_port,
        baudrate=SERIAL_BAUD,
        weights_pth="models/control_model_weights.pt",
    )
    state_machine = TargetStateMachine()

    # Setup Homography Projector
    dst_pts = np.array([[-70, 55], [70, 55], [70, -55], [-70, -55]], dtype=np.float32)
    projector = HomographyProjector(dst_pts)

    # 2. Setup Audio Rolling Buffer
    audio_buffer = np.zeros(AUDIO_WINDOW_SAMPLES, dtype=np.float32)
    audio_queue = queue.Queue()

    def audio_callback(indata, frames, time_info, status):
        audio_queue.put(indata.copy().reshape(-1))

    audio_stream = sd.InputStream(
        samplerate=AUDIO_SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=int(AUDIO_SAMPLE_RATE / TARGET_HZ),  # Fetch audio at 30Hz chunks
        callback=audio_callback,
    )

    # Audio state variables
    current_command = "hold"
    audio_labels = [
        "background",
        "go_red",
        "go_blue",
        "go_green",
        "go_yellow",
        "go_grey",
        "forward",
        "backward",
        "left",
        "right",
        "stop",
        "hold",
    ]  # Update to match your labels.json
    cand_label = None
    cand_count = 0

    # 3. Setup Camera
    camera = USBReceiver(camera_id=1)
    while camera.get_latest_frame() is None:
        time.sleep(0.1)
    print("Camera initialized.")

    print("\nSystem running! Enforcing 30Hz control loop...")

    with audio_stream:
        control.ser.reset_input_buffer()  # flush the homing prints / boot lines once

        # NEW: ball-loss tracking for the 3-second home-on-loss safety
        last_detected_time = time.perf_counter()
        homed_on_loss = False

        loop_count = 0  # NEW: drives the audio throttle

        while True:
            loop_start_time = time.perf_counter()
            loop_count += 1

            # --- A. PROCESS AUDIO (The Command) ---
            # Always drain the queue (cheap) so the rolling buffer stays current
            # and never backs up, even on loops where we skip inference.
            while not audio_queue.empty():
                new_audio = audio_queue.get()
                shift = len(new_audio)
                audio_buffer = np.roll(audio_buffer, -shift)
                audio_buffer[-shift:] = new_audio

            # CHANGED: only run the (expensive) audio CNN every Nth loop.
            if loop_count % AUDIO_EVERY == 0:
                audio_tensor = torch.tensor(audio_buffer).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = audio_model(audio_tensor)
                    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

                top_idx = int(np.argmax(probs))
                conf = probs[top_idx]
                detected_label = audio_labels[top_idx]

                # Debounce / Latch Logic (now counts *audio* frames, ~10 Hz)
                if conf > AUDIO_CONFIDENCE_THRESHOLD and detected_label != "background":
                    if detected_label == cand_label:
                        cand_count += 1
                    else:
                        cand_label = detected_label
                        cand_count = 1

                    if cand_count >= AUDIO_DEBOUNCE_FRAMES:
                        if current_command != cand_label:
                            print(f"🔊 Command Recognized: {cand_label}")
                        current_command = cand_label
                        cand_count = 0
                else:
                    cand_count = 0

            # --- B. PROCESS VISION (The Perception) ---
            frame = camera.get_latest_frame()
            cam_x, cam_y, marker_coords = None, None, None

            if frame is not None:
                cam_x, cam_y, marker_coords = process_vision_frame(
                    frame, yolo_model, mlp_corrector, projector, device
                )

            # --- C. PROCESS CONTROL (The Action) ---
            if cam_x is not None:
                # Ball is visible: refresh the loss timer and clear the homed flag.
                last_detected_time = loop_start_time
                homed_on_loss = False

                # 1. Ask state machine where the ball should go based on audio command
                state_machine.process_command(current_command, cam_x, cam_y)

                if marker_coords:
                    state_machine.update_markers(marker_coords)

                target_x, target_y = state_machine.get_target_coords()

                # Check if we reached the target and need to auto-hold
                state_machine.maybe_auto_hold(
                    cam_x, cam_y, marker_coords if marker_coords else {}
                )

                # 2. Feed physics data to the Control Policy.
                #    control.step() reads the STM32's echoed motor positions,
                #    runs the MLP, and sends the new target over serial.
                motor_targets = control.step(cam_x, cam_y, target_x, target_y)

            else:
                # Ball lost. Clear the velocity filter so reacquire doesn't spike.
                control.state.reset()

                # NEW: after a grace period, level/home the plate ONCE, mirroring
                # the old firmware rl_balance safety. The flag prevents spamming
                # "0,0,0" every frame (which would flood serial / backlog echoes).
                if (
                    not homed_on_loss
                    and (loop_start_time - last_detected_time) >= BALL_LOST_HOME_SEC
                ):
                    control.home()
                    homed_on_loss = True
                    print("[BALL LOST] No ball for 3s — homing plate to level.")

            # --- D. ENFORCE 30 Hz LOOP ---
            elapsed = time.perf_counter() - loop_start_time
            sleep_time = LOOP_TIME - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                print(f"⚠️ Loop missed 30Hz deadline! Took {elapsed*1000:.1f}ms")


if __name__ == "__main__":
    main()
