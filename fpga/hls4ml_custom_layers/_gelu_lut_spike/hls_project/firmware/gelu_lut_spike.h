#ifndef GELU_LUT_SPIKE_H_
#define GELU_LUT_SPIKE_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"


// Prototype of top level function for C-synthesis
void gelu_lut_spike(
    input_t x[8],
    result_t layer3_out[8]
);

// hls-fpga-machine-learning insert emulator-defines


#endif
