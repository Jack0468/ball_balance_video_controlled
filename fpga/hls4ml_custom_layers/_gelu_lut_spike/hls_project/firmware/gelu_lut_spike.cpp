#include <iostream>

#include "gelu_lut_spike.h"
#include "parameters.h"


void gelu_lut_spike(
    input_t x[8],
    result_t layer3_out[8]
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=x complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer3_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=x,layer3_out 
    #pragma HLS PIPELINE

    // hls-fpga-machine-learning insert load weights
#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        nnet::load_weights_from_txt<model_default_t, 64>(w2, "w2.txt");
        nnet::load_weights_from_txt<model_default_t, 8>(b2, "b2.txt");
        loaded_weights = true;    }
#endif
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    layer2_t layer2_out[8];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0

    nnet::dense<input_t, layer2_t, config2>(x, layer2_out, w2, b2); // fc

    nnet::gelu_lut<layer2_t, config3>(layer2_out, layer3_out); // act

}

