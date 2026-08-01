import os
import threading
import queue
import time
import numpy as np
import sounddevice as sd
import onnxruntime as ort

try:
    import soundfile as sf
    _HAS_SOUNDFILE = True
except ImportError:
    _HAS_SOUNDFILE = False
    try:
        from scipy.io import wavfile
    except ImportError:
        wavfile = None

SAMPLE_RATE = 16_000
MODEL_WINDOW_SECONDS = 1.25
OUTPUT_SEQUENCE_LENGTH = int(SAMPLE_RATE * MODEL_WINDOW_SECONDS)
N_FFT = 255
HOP_LENGTH = 128

LABEL_NAMES = ["_background_", "go_blue", "go_green", "go_red", "go_yellow", "hold", "stop", "go_grey", "forward", "backward", "left", "right"]

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

def waveform_to_spectrogram_np(waveform):
    # This recreates the PyTorch STFT behavior using numpy/scipy if possible
    # Wait, the PyTorch implementation uses torch.stft which is quite specific.
    # We can just import torch and use it for the STFT to avoid discrepancies.
    import torch
    torch.set_num_threads(1)
    waveform_pt = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)
    window = torch.hann_window(N_FFT)
    spec = torch.stft(waveform_pt, n_fft=N_FFT, hop_length=HOP_LENGTH, window=window, return_complex=True, center=False)
    
    spec = spec.abs()
    # Ignoring noise profile subtraction for this ONNX version for simplicity,
    # or we can pass it if we want. We'll skip it for now unless needed.
        
    spec = spec.transpose(1, 2)
    spec = torch.log(spec + 1e-6)
    return spec.numpy() # Convert back to numpy

class AudioCommandReceiverONNX:
    def __init__(self, model_path, step_seconds=0.2, source_file=None):
        print(f"Loading Audio Model ONNX from {model_path}...")
        
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        
        self.session = ort.InferenceSession(model_path, sess_options=opts, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        
        self.step_seconds = step_seconds
            
        self.window_samples = OUTPUT_SEQUENCE_LENGTH
        self.step_samples = int(SAMPLE_RATE * self.step_seconds)
        self.audio_buffer = np.zeros(self.window_samples, dtype=np.float32)
        
        self.command_queue = queue.Queue(maxsize=1)
        self.running = True
        
        self.chunk_queue = queue.Queue()
        self.latest_inference_time_ms = 0.0
        
        self.min_confidence = 0.8
        self.min_margin = 0.15
        self.last_pushed_command = None
        
        self.source_file = source_file
        if self.source_file:
            print(f"Audio receiver initialized on File Stream: {self.source_file}")
            self.thread_file = threading.Thread(target=self._file_reader_loop, daemon=True)
            self.thread_file.start()
        else:
            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=self.step_samples,
                callback=self._audio_callback
            )
            self.stream.start()
            print("Audio receiver initialized on laptop microphone.")
        
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()

    def _file_reader_loop(self):
        try:
            if _HAS_SOUNDFILE:
                audio, sr = sf.read(self.source_file)
            elif wavfile is not None:
                sr, audio = wavfile.read(self.source_file)
                audio = audio.astype(np.float32)
            else:
                raise RuntimeError("No supported audio file reader is installed. Install soundfile or scipy.")

            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            
            if sr != SAMPLE_RATE:
                print(f"Resampling audio from {sr} Hz to {SAMPLE_RATE} Hz...")
                audio = self._resample_audio(audio, sr, SAMPLE_RATE)
            
            idx = 0
            while self.running and idx < len(audio):
                start_t = time.perf_counter()
                end_idx = min(idx + self.step_samples, len(audio))
                chunk = audio[idx:end_idx].astype(np.float32)
                
                if len(chunk) < self.step_samples:
                    chunk = np.pad(chunk, (0, self.step_samples - len(chunk)))
                    
                self.chunk_queue.put(chunk)
                idx += self.step_samples
                
                elapsed = time.perf_counter() - start_t
                sleep_time = self.step_seconds - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
            print("Audio file stream completed.")
        except Exception as e:
            print(f"Error in file reader loop: {e}")

    def _resample_audio(self, audio, src_sr, dst_sr):
        try:
            import resampy
            return resampy.resample(audio, src_sr, dst_sr)
        except ImportError:
            print("Warning: resampy not installed, attempting scipy.signal.resample")
            try:
                from scipy.signal import resample
                num_samples = int(len(audio) * dst_sr / src_sr)
                return resample(audio, num_samples).astype(np.float32)
            except Exception as e:
                raise RuntimeError(f"Audio resampling failed: {e}")

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            pass
        self.chunk_queue.put(indata.copy().squeeze())
        
    def _process_loop(self):
        def softmax(x):
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum(axis=-1, keepdims=True)

        while self.running:
            try:
                new_chunk = self.chunk_queue.get(timeout=1.0)
            except queue.Empty:
                continue
                
            self.audio_buffer = np.roll(self.audio_buffer, -len(new_chunk))
            self.audio_buffer[-len(new_chunk):] = new_chunk
            
            aligned = align_speech_to_fixed_length(self.audio_buffer)
            if aligned is None:
                self.last_pushed_command = None
            else:
                inf_start = time.perf_counter()
                spec_np = waveform_to_spectrogram_np(aligned)
                
                # ONNX inference
                logits = self.session.run(None, {self.input_name: spec_np})[0][0]
                probs = softmax(logits)
                self.latest_inference_time_ms = (time.perf_counter() - inf_start) * 1000.0
                
                top_id = int(np.argmax(probs))
                top_label = LABEL_NAMES[top_id]
                top_conf = float(probs[top_id])
                
                top_two = np.partition(probs, -2)[-2:]
                margin = float(top_two[-1] - top_two[-2])
                
                # --- DEBUG PRINT ---
                print(f"[AUDIO DEBUG] Pred: {top_label} | Conf: {top_conf:.2f} | Margin: {margin:.2f}")
                
                if top_conf >= self.min_confidence and margin >= self.min_margin:
                    if top_label != "_background_" and top_label != self.last_pushed_command:
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
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
