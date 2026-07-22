#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Inference entrypoint: input is 64x64 spectrogram (single channel)
// output is logits array of length 6 (number of labels)
void audio_inference(const float input[64][64], float output[6]);

#ifdef __cplusplus
}
#endif
