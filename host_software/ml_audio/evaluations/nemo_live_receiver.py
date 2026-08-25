"""NeMo-model live receiver -- mirrors AudioCommandReceiver's threading/
buffer/gating harness (audio_receiver_pytorch.py) exactly, so live-stream
results are directly comparable between the custom CNN and the pretrained-
backbone track (see docs/plans/audio_eval_notebook_refactor_plan.md,
"Pretrained Backbone / Transfer Learning Track").

Kept as a separate, self-contained class rather than refactoring the
production AudioCommandReceiver -- this track is still exploratory, not yet
a deployment decision, and audio_receiver_pytorch.py is used elsewhere
(realtime_audio_inference.py) where destabilizing it for a not-yet-adopted
second model family isn't worth the risk.

Same window size, step interval, and confidence/margin gating thresholds as
AudioCommandReceiver -- a fair comparison needs identical harness behavior,
only the model-specific preprocessing/forward pass differs. NeMo's own
preprocessor (MFCC features, computed inside model.forward() -- see the
Phase 0 finding that the exported ONNX graph takes precomputed features,
not raw waveform) replaces audio_dsp.py's waveform_to_spectrogram here,
since that transform is specific to the custom CNN's architecture.

NeMo import is deferred to __init__ (not module level) so this file stays
importable -- including into prepare_colab_package.py's bundle and any
local syntax/logic check -- without NeMo installed, matching how audio_dsp.py
was split out to avoid an unnecessary sounddevice dependency for the custom
CNN's track. This module itself needs no NeMo-specific or Windows-hostile
dependencies to import, only to instantiate.
"""

import queue
import threading
import time

import numpy as np
import soundfile as sf
import torch

from ml_audio.audio_dsp import OUTPUT_SEQUENCE_LENGTH, SAMPLE_RATE


class NemoAudioCommandReceiver:
    def __init__(
        self,
        nemo_model_path: str,
        step_seconds: float = 0.2,
        source_file: str | None = None,
        min_confidence: float = 0.8,
        min_margin: float = 0.15,
    ):
        import nemo.collections.asr as nemo_asr

        print(f"Loading NeMo model from {nemo_model_path}...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = nemo_asr.models.EncDecClassificationModel.restore_from(nemo_model_path)
        self.model.to(self.device)
        self.model.eval()
        # model.labels (the instance attribute) is only populated as a side
        # effect of setup_training_data()/setup_validation_data() -- it's
        # None after a plain restore_from() with no training setup, even
        # though the checkpoint's real label order is right there in
        # model.cfg.labels (confirmed by comparing both directly: cfg.labels,
        # cfg.train_ds.labels, and cfg.validation_ds.labels all agree, and
        # match the decoder's num_classes). Caught locally before this ever
        # ran against real audio.
        self.labels = list(self.model.cfg.labels)
        print(f"Loaded on {self.device}. Labels ({len(self.labels)}): {self.labels}")

        self.step_seconds = step_seconds
        self.window_samples = OUTPUT_SEQUENCE_LENGTH
        self.step_samples = int(SAMPLE_RATE * step_seconds)
        self.audio_buffer = np.zeros(self.window_samples, dtype=np.float32)

        self.command_queue = queue.Queue(maxsize=1)
        self.running = True
        self.chunk_queue = queue.Queue()

        self.min_confidence = min_confidence
        self.min_margin = min_margin
        self.last_pushed_command = None

        self.source_file = source_file
        if not self.source_file:
            raise NotImplementedError(
                "NemoAudioCommandReceiver only supports source_file playback mode "
                "(this track hasn't been wired to a live microphone yet)."
            )
        print(f"NeMo audio receiver initialized on File Stream: {self.source_file}")
        self.thread_file = threading.Thread(target=self._file_reader_loop, daemon=True)
        self.thread_file.start()

        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()

    def _file_reader_loop(self):
        try:
            audio, sr = sf.read(self.source_file)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)

            if sr != SAMPLE_RATE:
                print(f"Resampling audio from {sr} Hz to {SAMPLE_RATE} Hz...")
                from scipy.signal import resample
                num_samples = int(len(audio) * SAMPLE_RATE / sr)
                audio = resample(audio, num_samples).astype(np.float32)

            idx = 0
            while self.running and idx < len(audio):
                start_t = time.perf_counter()
                end_idx = min(idx + self.step_samples, len(audio))
                chunk = audio[idx:end_idx].astype(np.float32)

                if len(chunk) < self.step_samples:
                    chunk = np.pad(chunk, (0, self.step_samples - len(chunk)))

                self.chunk_queue.put(chunk)
                idx += self.step_samples

                # Simulate real-time stream cadence, same as AudioCommandReceiver.
                elapsed = time.perf_counter() - start_t
                sleep_time = self.step_seconds - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
            print("Audio file stream completed.")
        except Exception as e:
            print(f"Error in file reader loop: {e}")

    def _process_loop(self):
        while self.running:
            try:
                new_chunk = self.chunk_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            self.audio_buffer = np.roll(self.audio_buffer, -len(new_chunk))
            self.audio_buffer[-len(new_chunk):] = new_chunk

            audio_t = torch.as_tensor(
                self.audio_buffer, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            len_t = torch.tensor([self.window_samples], device=self.device)
            with torch.no_grad():
                logits = self.model(input_signal=audio_t, input_signal_length=len_t)
                probs = torch.nn.functional.softmax(logits, dim=-1)[0].cpu().numpy()

            top_id = int(np.argmax(probs))
            top_label = self.labels[top_id]
            top_conf = float(probs[top_id])

            top_two = np.partition(probs, -2)[-2:]
            margin = float(top_two[-1] - top_two[-2])

            if top_conf >= self.min_confidence and margin >= self.min_margin:
                if top_label != self.last_pushed_command:
                    if self.command_queue.full():
                        try:
                            self.command_queue.get_nowait()
                        except queue.Empty:
                            pass
                    self.command_queue.put(top_label)
                    self.last_pushed_command = top_label
            else:
                self.last_pushed_command = None

    def get_latest_command(self):
        try:
            return self.command_queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        self.running = False
