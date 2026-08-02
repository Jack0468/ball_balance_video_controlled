import openvino as ov
import numpy as np
import time

LABEL_NAMES = ["go_blue", "go_green", "go_red", "go_yellow", "hold", "stop"]


class OpenVINOPipeline:
    def __init__(self, yolo_xml, audio_xml, mlp_corrector_iphone_v1_xml, device="CPU", jobs=1):
        self.core = ov.Core()

        print(f"Loading OpenVINO models for device {device}...")
        self.yolo_model = self.core.read_model(yolo_xml)

        if audio_xml:
            self.audio_model = self.core.read_model(audio_xml)
            self.audio_compiled = self.core.compile_model(self.audio_model, device)
            self.audio_queue = ov.AsyncInferQueue(self.audio_compiled, jobs)
            self.audio_queue.set_callback(self._audio_callback)
        else:
            self.audio_model = None
            self.audio_compiled = None
            self.audio_queue = None

        if mlp_corrector_iphone_v1_xml:
            self.mlp_corrector_iphone_v1_model = self.core.read_model(mlp_corrector_iphone_v1_xml)
            self.mlp_corrector_iphone_v1_compiled = self.core.compile_model(
                self.mlp_corrector_iphone_v1_model, device
            )
            self.mlp_corrector_iphone_v1_queue = ov.AsyncInferQueue(
                self.mlp_corrector_iphone_v1_compiled, jobs
            )
            self.mlp_corrector_iphone_v1_queue.set_callback(self._mlp_corrector_iphone_v1_callback)
        else:
            self.mlp_corrector_iphone_v1_model = None
            self.mlp_corrector_iphone_v1_compiled = None
            self.mlp_corrector_iphone_v1_queue = None

        self.yolo_compiled = self.core.compile_model(self.yolo_model, device)
        self.yolo_queue = ov.AsyncInferQueue(self.yolo_compiled, jobs)
        self.yolo_queue.set_callback(self._yolo_callback)

        # Shared state, thread-safe due to Python GIL only protecting the dict access
        self.state = {
            "yolo_result": None,  # Raw output tensor
            "audio_command": None,  # String command (debounced)
            "mlp_corrector_iphone_v1_output": None,  # Tuple of (x, y)
        }

        # Audio debouncing variables
        self.audio_history = []
        # Conservative thresholds to reduce false motor-noise triggers.
        self.min_confidence = 0.72
        self.min_margin = 0.18
        self.command_cooldown_seconds = 5.0
        self.last_command_time = -float("inf")
        self.ready_for_new_command = True
        self.silence_frames = 0
        self.required_silence_frames = 2

    def _yolo_callback(self, infer_request, user_data):
        try:
            # YOLOv8 OpenVINO output
            # Output node is generally the first one
            res = infer_request.get_output_tensor(0).data
            # For YOLOv8, it's (1, 84, 8400) usually
            self.state["yolo_result"] = res[0].copy()
        except Exception as e:
            print(f"YOLO Callback Error: {e}")

    def _audio_callback(self, infer_request, user_data):
        try:
            probs = infer_request.get_output_tensor(0).data[0].copy()
            # Softmax just in case
            exp_probs = np.exp(probs - np.max(probs))
            probs = exp_probs / np.sum(exp_probs)

            top_id = int(np.argmax(probs))
            top_label = LABEL_NAMES[top_id]
            top_conf = float(probs[top_id])

            top_two = np.partition(probs, -2)[-2:]
            margin = float(top_two[-1] - top_two[-2])

            if top_conf >= self.min_confidence and margin >= self.min_margin:
                self.silence_frames = 0
                self.audio_history.append(top_label)
            else:
                self.audio_history.append(None)
                self.silence_frames += 1
                if self.silence_frames >= self.required_silence_frames:
                    self.ready_for_new_command = True

            if len(self.audio_history) > 3:
                self.audio_history.pop(0)

            if len(self.audio_history) == 3:
                p1, p2, p3 = self.audio_history
                cooldown_elapsed = (
                    time.time() - self.last_command_time
                ) >= self.command_cooldown_seconds
                if (
                    p2 is not None
                    and p2 == p3
                    and p1 != p2
                    and self.ready_for_new_command
                    and cooldown_elapsed
                ):
                    self.state["audio_command"] = p2
                    self.last_command_time = time.time()
                    self.ready_for_new_command = False
        except Exception as e:
            print(f"Audio Callback Error: {e}")

    def _mlp_corrector_iphone_v1_callback(self, infer_request, user_data):
        try:
            res = infer_request.get_output_tensor(0).data[0]
            self.state["mlp_corrector_iphone_v1_output"] = (float(res[0]), float(res[1]))
        except Exception as e:
            print(f"Corrector Callback Error: {e}")

    # --- Dispatchers ---
    def dispatch_yolo(self, frame_preprocessed):
        """Dispatches an NCHW normalized frame if queue is ready."""
        if self.yolo_queue.is_ready():
            self.yolo_queue.start_async({0: frame_preprocessed})

    def dispatch_audio(self, spectrogram):
        if self.audio_queue.is_ready():
            self.audio_queue.start_async({0: spectrogram})

    def register_audio_silence(self):
        self.audio_history.append(None)
        if len(self.audio_history) > 3:
            self.audio_history.pop(0)
        self.silence_frames += 1
        if self.silence_frames >= self.required_silence_frames:
            self.ready_for_new_command = True

    def dispatch_mlp_corrector_iphone_v1(self, features):
        """Features should be shape (1, 14)"""
        if self.mlp_corrector_iphone_v1_queue.is_ready():
            self.mlp_corrector_iphone_v1_queue.start_async({0: features})

    def get_and_clear_audio_command(self):
        cmd = self.state["audio_command"]
        if cmd is not None:
            self.state["audio_command"] = None
        return cmd
