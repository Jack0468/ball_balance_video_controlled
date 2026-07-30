import sounddevice as sd
import queue
import time
import numpy as np

print("Opening audio...")
try:
    audio_stream = sd.InputStream(
        samplerate=16000, channels=1, dtype="float32",
        blocksize=16000, callback=lambda *args: None
    )
    audio_stream.start()
    print("Audio started.")
    time.sleep(2)
    audio_stream.stop()
    print("Audio stopped.")
except Exception as e:
    print("Audio error:", e)
