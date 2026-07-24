import threading
import queue
import time
import argparse
import sounddevice as sd
import numpy as np
import torch

try:
    from .audio_pytorch_runtime import (
        LABEL_NAMES,
        OUTPUT_SEQUENCE_LENGTH,
        SAMPLE_RATE,
        align_speech_to_fixed_length,
        default_model_path,
        load_pytorch_audio_model,
        predict_probabilities,
    )
except ImportError:
    from audio_pytorch_runtime import (
        LABEL_NAMES,
        OUTPUT_SEQUENCE_LENGTH,
        SAMPLE_RATE,
        align_speech_to_fixed_length,
        default_model_path,
        load_pytorch_audio_model,
        predict_probabilities,
    )

class AudioCommandReceiver:
    def __init__(self, model_path=None, step_seconds=0.2, command_cooldown_seconds=5.0):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model, resolved_model_path = load_pytorch_audio_model(model_path=model_path, device=self.device)
        print(f"Loading Audio Model from {resolved_model_path}...")
        self.step_seconds = step_seconds
        self.command_cooldown_seconds = command_cooldown_seconds
        
        self.window_samples = OUTPUT_SEQUENCE_LENGTH
        self.step_samples = int(SAMPLE_RATE * self.step_seconds)
        self.audio_buffer = np.zeros(self.window_samples, dtype=np.float32)
        
        self.command_queue = queue.Queue(maxsize=1)
        self.running = True
        self.last_command_time = -float("inf")
        self.ready_for_new_command = True
        self.latest_status = {
            "timestamp": time.time(),
            "top_label": None,
            "top_confidence": 0.0,
            "margin": 0.0,
            "accepted": False,
            "reason": "starting",
        }
        
        # We need a small queue to pass chunks from the audio callback to the processing thread
        self.chunk_queue = queue.Queue()
        
        # Configuration for debouncing
        self.min_confidence = 0.6
        self.min_margin = 0.10
        self.recent_predictions = []
        
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=self.step_samples,
            callback=self._audio_callback
        )
        self.stream.start()
        
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()
        print("Audio receiver initialized on laptop microphone.")

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            pass # ignore warnings for now
        self.chunk_queue.put(indata.copy().squeeze())
        
    def _process_loop(self):
        while self.running:
            try:
                # Wait for the next block of audio
                new_chunk = self.chunk_queue.get(timeout=1.0)
            except queue.Empty:
                continue
                
            # Shift buffer left and append new chunk
            self.audio_buffer = np.roll(self.audio_buffer, -len(new_chunk))
            self.audio_buffer[-len(new_chunk):] = new_chunk
            
            aligned, meta = align_speech_to_fixed_length(self.audio_buffer)
            if aligned is None:
                self.ready_for_new_command = True
                self.latest_status = {
                    "timestamp": time.time(),
                    "top_label": None,
                    "top_confidence": 0.0,
                    "margin": 0.0,
                    "accepted": False,
                    "reason": meta.get("reason", "unclassified"),
                }
                self.recent_predictions.append(None)
            else:
                probs = predict_probabilities(self.model, aligned, device=self.device)
                
                top_id = int(np.argmax(probs))
                top_label = LABEL_NAMES[top_id]
                top_conf = float(probs[top_id])
                
                top_two = np.partition(probs, -2)[-2:]
                margin = float(top_two[-1] - top_two[-2])
                accepted = top_conf >= self.min_confidence and margin >= self.min_margin
                now = time.time()
                cooldown_remaining = max(0.0, self.command_cooldown_seconds - (now - self.last_command_time))
                self.latest_status = {
                    "timestamp": now,
                    "top_label": top_label,
                    "top_confidence": top_conf,
                    "margin": margin,
                    "accepted": accepted,
                    "reason": "accepted" if accepted else "below_threshold",
                    "cooldown_remaining": cooldown_remaining,
                    "ready_for_new_command": self.ready_for_new_command,
                }
                
                if accepted:
                    self.recent_predictions.append(top_label)
                else:
                    self.recent_predictions.append(None)
                    
            # Debounce: require 2 identical consecutive valid predictions to trigger a command
            if len(self.recent_predictions) > 3:
                self.recent_predictions.pop(0)
                
            if len(self.recent_predictions) == 3:
                p1, p2, p3 = self.recent_predictions
                now = time.time()
                cooldown_elapsed = (now - self.last_command_time) >= self.command_cooldown_seconds
                if (
                    p2 is not None
                    and p2 == p3
                    and p1 != p2
                    and self.ready_for_new_command
                    and cooldown_elapsed
                ):
                    # Output command!
                    if self.command_queue.full():
                        try:
                            self.command_queue.get_nowait()
                        except queue.Empty:
                            pass
                    self.command_queue.put(p2)
                    self.last_command_time = now
                    self.ready_for_new_command = False

    def get_latest_command(self):
        try:
            return self.command_queue.get_nowait()
        except queue.Empty:
            return None

    def get_latest_status(self):
        return dict(self.latest_status)

    def stop(self):
        self.running = False
        self.stream.stop()
        self.stream.close()


def main():
    parser = argparse.ArgumentParser(description="Realtime microphone test for the PyTorch audio command receiver.")
    parser.add_argument(
        "--model-path",
        default=str(default_model_path()),
        help="Path to the exported PyTorch checkpoint (.pth).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="How many seconds to listen before stopping. Use 0 for continuous mode.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=0.2,
        help="How often to poll the receiver internally.",
    )
    parser.add_argument(
        "--report-seconds",
        type=float,
        default=2.0,
        help="How often to print one command/status result while the mic stays on.",
    )
    parser.add_argument(
        "--command-gap-seconds",
        type=float,
        default=5.0,
        help="Minimum time gap between accepted commands.",
    )
    args = parser.parse_args()

    receiver = AudioCommandReceiver(
        model_path=args.model_path,
        command_cooldown_seconds=args.command_gap_seconds,
    )
    if args.duration > 0:
        print(f"Listening for {args.duration:.1f}s. Reporting every {args.report_seconds:.1f}s.")
    else:
        print(f"Listening continuously. Reporting every {args.report_seconds:.1f}s. Press Ctrl+C to stop.")

    start = time.time()
    next_report_time = start + args.report_seconds
    detected_in_window = []
    last_status = receiver.get_latest_status()
    try:
        while True:
            now = time.time()
            elapsed = now - start
            if args.duration > 0 and elapsed >= args.duration:
                break

            latest = receiver.get_latest_command()
            status = receiver.get_latest_status()
            last_status = status
            if latest is not None:
                detected_in_window.append(latest)

            if now >= next_report_time:
                if detected_in_window:
                    reported_command = detected_in_window[-1]
                    print(f"predicted {reported_command}")
                detected_in_window.clear()
                next_report_time += args.report_seconds

            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        print("Stopping microphone listener...")
    finally:
        receiver.stop()
        print("Audio receiver stopped.")


if __name__ == "__main__":
    main()
