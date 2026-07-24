import numpy as np

def numpy_stft(waveform, n_fft=255, hop_length=128):
    # TensorFlow uses periodic Hann window by default: `tf.signal.hann_window`
    # Periodic Hann is np.hanning(n_fft+1)[:-1]
    window = np.hanning(n_fft + 1)[:-1].astype(np.float32)
    
    # Frame the signal
    num_frames = 1 + (len(waveform) - n_fft) // hop_length
    frames = np.lib.stride_tricks.as_strided(
        waveform, 
        shape=(num_frames, n_fft), 
        strides=(waveform.strides[0] * hop_length, waveform.strides[0])
    )
    
    # Apply window
    windowed_frames = frames * window
    
    # RFFT (matches tf.signal.stft which pads to enclosing power of 2)
    # n_fft=255, smallest enclosing power of 2 is 256
    fft_length = 2**(n_fft - 1).bit_length()
    stft_matrix = np.fft.rfft(windowed_frames, n=fft_length, axis=-1)
    
    spec = np.abs(stft_matrix)
    spec = np.log(spec + 1e-6)
    
    # Add batch and channel dims: shape (1, num_frames, bins, 1)
    return spec[np.newaxis, ..., np.newaxis].astype(np.float32)
