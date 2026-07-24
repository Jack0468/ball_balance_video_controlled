import threading
import queue
import time
import numpy as np
import sounddevice as sd
import torch
from ml_audio.audio_command_classifier_pytorch import AudioCommandClassifier

SAMPLE_RATE = 16_000
MODEL_WINDOW_SECONDS = 1.25
OUTPUT_SEQUENCE_LENGTH = int(SAMPLE_RATE * MODEL_WINDOW_SECONDS)
N_FFT = 255
HOP_LENGTH = 128

LABEL_NAMES = ["go_blue", "go_green", "go_red", "go_yellow", "hold", "stop"]

def align_speech_to_fixed_length(audio, target_samples=OUTPUT_SEQUENCE_LENGTH):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    peak = np.max(np.abs(audio))
    rms = np.sqrt(np.mean(audio ** 2))

    if peak < 0.03 or rms < 0.003:
        return None

    threshold = max(0.015, peak * 0.08)
    active = np.where(np.abs(audio) > threshold)[0]
    if len(active) == 0:
        return None

    start = max(0, active[0] - int(0.08 * SAMPLE_RATE))
    end = min(len(audio), active[-1] + int(0.12 * SAMPLE_RATE))
    audio = audio[start:end]

    if len(audio) > target_samples:
        audio = audio[:target_samples]
    if len(audio) < target_samples:
        audio = np.pad(audio, (0, target_samples - len(audio)))

    peak = np.max(np.abs(audio))
    if peak > 1e-6:
        audio = audio / peak * 0.95

    return audio.astype(np.float32)

def waveform_to_spectrogram(waveform):
    waveform_pt = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)
    window = torch.hann_window(N_FFT)
    spec = torch.stft(waveform_pt, n_fft=N_FFT, hop_length=HOP_LENGTH, window=window, return_complex=True)
    spec = spec.abs()
    # PyTorch stft returns [batch, freqs, frames]. TF returns [batch, frames, freqs].
    spec = spec.transpose(1, 2)
    spec = torch.log(spec + 1e-6)
    return spec

class AudioCommandReceiver:
    def __init__(self, model_path, step_seconds=0.2):
        print(f"Loading Audio Model from {model_path}...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AudioCommandClassifier()
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        self.step_seconds = step_seconds
        
        self.window_samples = OUTPUT_SEQUENCE_LENGTH
        self.step_samples = int(SAMPLE_RATE * self.step_seconds)
        self.audio_buffer = np.zeros(self.window_samples, dtype=np.float32)
        
        self.command_queue = queue.Queue(maxsize=1)
        self.running = True
        
        self.chunk_queue = queue.Queue()
        
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
            pass
        self.chunk_queue.put(indata.copy().squeeze())
        
    def _process_loop(self):
        while self.running:
            try:
                new_chunk = self.chunk_queue.get(timeout=1.0)
            except queue.Empty:
                continue
                
            self.audio_buffer = np.roll(self.audio_buffer, -len(new_chunk))
            self.audio_buffer[-len(new_chunk):] = new_chunk
            
            aligned = align_speech_to_fixed_length(self.audio_buffer)
            if aligned is None:
                self.recent_predictions.append(None)
            else:
                spec = waveform_to_spectrogram(aligned).to(self.device)
                with torch.no_grad():
                    logits = self.model(spec)
                    probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()[0]
                
                top_id = int(np.argmax(probs))
                top_label = LABEL_NAMES[top_id]
                top_conf = float(probs[top_id])
                
                top_two = np.partition(probs, -2)[-2:]
                margin = float(top_two[-1] - top_two[-2])
                
                if top_conf >= self.min_confidence and margin >= self.min_margin:
                    self.recent_predictions.append(top_label)
                else:
                    self.recent_predictions.append(None)
                    
            if len(self.recent_predictions) > 3:
                self.recent_predictions.pop(0)
                
            if len(self.recent_predictions) == 3:
                p1, p2, p3 = self.recent_predictions
                if p2 is not None and p2 == p3 and p1 != p2: 
                    if self.command_queue.full():
                        try:
                            self.command_queue.get_nowait()
                        except queue.Empty:
                            pass
                    self.command_queue.put(p2)

    def get_latest_command(self):
        try:
            return self.command_queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        self.running = False
        self.stream.stop()
        self.stream.close()
