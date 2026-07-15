import threading
import time
import numpy as np
import sounddevice as sd
import tensorflow as tf
import os

SAMPLE_RATE = 16_000
CLIP_SECONDS = 1.25
OUTPUT_SEQUENCE_LENGTH = int(SAMPLE_RATE * CLIP_SECONDS)

COMMANDS = [
    "go_red",
    "go_blue",
    "go_green",
    "go_yellow",
    "hold",
    "stop",
]

COMMAND_PHRASES = {
    "go_red": "go red",
    "go_blue": "go blue",
    "go_green": "go green",
    "go_yellow": "go yellow",
    "hold": "hold",
    "stop": "stop",
}

MIN_CONFIDENCE = 0.70
STOP_CONFIDENCE = 0.55

ACTION_MAP = {
    "go_red": {"mode": "colour_select", "target_colour": "red", "hold": False, "emergency_stop": False},
    "go_blue": {"mode": "colour_select", "target_colour": "blue", "hold": False, "emergency_stop": False},
    "go_green": {"mode": "colour_select", "target_colour": "green", "hold": False, "emergency_stop": False},
    "go_yellow": {"mode": "colour_select", "target_colour": "yellow", "hold": False, "emergency_stop": False},
    "hold": {"mode": "hold", "target_colour": None, "hold": True, "emergency_stop": False},
    "stop": {"mode": "stop", "target_colour": None, "hold": True, "emergency_stop": True},
}

def align_speech_to_fixed_length(audio, target_samples=OUTPUT_SEQUENCE_LENGTH):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    peak = np.max(np.abs(audio))
    rms = np.sqrt(np.mean(audio ** 2))

    if peak < 0.03 or rms < 0.003:
        return None, {"reason": "too_quiet", "peak": float(peak), "rms": float(rms)}

    threshold = max(0.015, peak * 0.08)
    active = np.where(np.abs(audio) > threshold)[0]

    if len(active) == 0:
        return None, {"reason": "no_speech_detected"}

    start = active[0]
    end = active[-1]

    pre_roll = int(0.08 * SAMPLE_RATE)
    post_roll = int(0.12 * SAMPLE_RATE)

    start = max(0, start - pre_roll)
    end = min(len(audio), end + post_roll)

    audio = audio[start:end]

    if len(audio) > target_samples:
        audio = audio[:target_samples]
    if len(audio) < target_samples:
        audio = np.pad(audio, (0, target_samples - len(audio)))

    peak = np.max(np.abs(audio))
    if peak > 1e-6:
        audio = audio / peak * 0.95

    return audio.astype(np.float32), {"reason": "ok"}

def get_spectrogram(waveform):
    spectrogram = tf.signal.stft(waveform, frame_length=512, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.math.log(spectrogram + 1e-6)
    return spectrogram[..., tf.newaxis]

class AudioListener:
    def __init__(self, model_path=None):
        self.running = False
        self.buffer = np.zeros(OUTPUT_SEQUENCE_LENGTH, dtype=np.float32)
        self.lock = threading.Lock()
        
        self.latest_state = dict(ACTION_MAP["hold"])
        self.latest_state["reason"] = "initializing"
        
        # Load model
        if model_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(os.path.dirname(script_dir), "models", "audio_command_classifier", "best_classifier.keras")
        
        print("Loading audio model from:", model_path)
        self.model = tf.keras.models.load_model(model_path)
        self.label_names = np.array(COMMANDS)
        
        self.stream = None
        self.inference_thread = None

    def _audio_callback(self, indata, frames, time_info, status):
        with self.lock:
            # Shift buffer left and append new data
            self.buffer[:-frames] = self.buffer[frames:]
            self.buffer[-frames:] = np.squeeze(indata, axis=-1)

    def _inference_loop(self):
        while self.running:
            time.sleep(0.5) # Run inference every 0.5s to avoid freezing
            
            with self.lock:
                waveform = self.buffer.copy()
            
            # Predict
            audio_aligned, info = align_speech_to_fixed_length(waveform, target_samples=OUTPUT_SEQUENCE_LENGTH)
            if audio_aligned is None:
                continue # Too quiet or no speech
                
            waveform_tf = tf.convert_to_tensor(audio_aligned, dtype=tf.float32)[tf.newaxis, :]
            spec = get_spectrogram(waveform_tf)
            logits = self.model(spec, training=False)
            probabilities = tf.nn.softmax(logits, axis=-1).numpy()[0]
            
            class_id = int(np.argmax(probabilities))
            command = str(self.label_names[class_id])
            confidence = float(probabilities[class_id])
            
            # Update target state
            phrase = COMMAND_PHRASES.get(command, command)
            if command == "stop" and confidence >= STOP_CONFIDENCE:
                state = dict(ACTION_MAP["stop"])
            elif confidence < MIN_CONFIDENCE:
                # Do not overwrite if low confidence, just keep previous state
                continue
            else:
                state = dict(ACTION_MAP[command])
                
            state.update({
                "source": "audio_module",
                "command": command,
                "phrase": phrase,
                "confidence": confidence,
                "timestamp": time.time(),
            })
            
            with self.lock:
                self.latest_state = state
                
    def start(self):
        self.running = True
        self.stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', callback=self._audio_callback)
        self.stream.start()
        
        self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.inference_thread.start()
        print("Audio listener started in background.")
        
    def stop(self):
        self.running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
        if self.inference_thread:
            self.inference_thread.join()
            
    def get_latest_command(self):
        with self.lock:
            return dict(self.latest_state)
