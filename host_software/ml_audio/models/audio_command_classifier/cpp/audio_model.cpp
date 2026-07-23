#include "audio_model_weights.h"
#include <algorithm>
#include <cmath>
#include <cstdio>

static const int MAX_PIX = 64 * 64;

static inline float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

static inline size_t conv_index(
    int kh, int kw, int in_ch, int out_ch,
    int k, int l, int ic, int oc
) {
    return (((size_t)k * kw + l) * in_ch + ic) * out_ch + oc;
}

static void normalize_input_inplace(float in_map[][MAX_PIX], int h, int w) {
    const float mean = NORMALIZATION_MEAN[0];
    const float variance = NORMALIZATION_VARIANCE[0];
    const float denom = std::sqrt(variance + NORMALIZATION_EPSILON);

    for (int i = 0; i < h * w; ++i) {
        in_map[0][i] = (in_map[0][i] - mean) / denom;
    }
}

static void conv2d_same_map(
    const float in_map[][MAX_PIX],
    int in_h,
    int in_w,
    int in_ch,
    const float *kernel,
    int kh,
    int kw,
    int out_ch,
    const float *biases,
    float out_map[][MAX_PIX]
) {
    int pad_h = kh / 2;
    int pad_w = kw / 2;
    int hw = in_h * in_w;

    for (int oc = 0; oc < out_ch; ++oc) {
        float b = biases ? biases[oc] : 0.0f;
        for (int i = 0; i < hw; ++i) {
            out_map[oc][i] = b;
        }
    }

    for (int y = 0; y < in_h; ++y) {
        for (int x = 0; x < in_w; ++x) {
            for (int k = 0; k < kh; ++k) {
                int in_y = y + k - pad_h;
                if (in_y < 0 || in_y >= in_h) continue;

                for (int l = 0; l < kw; ++l) {
                    int in_x = x + l - pad_w;
                    if (in_x < 0 || in_x >= in_w) continue;

                    for (int ic = 0; ic < in_ch; ++ic) {
                        float in_val = in_map[ic][in_y * in_w + in_x];

                        for (int oc = 0; oc < out_ch; ++oc) {
                            size_t idx = conv_index(kh, kw, in_ch, out_ch, k, l, ic, oc);
                            out_map[oc][y * in_w + x] += in_val * kernel[idx];
                        }
                    }
                }
            }
        }
    }
}

static void batchnorm_inplace(
    float map[][MAX_PIX],
    int h,
    int w,
    int channels,
    const float *gamma,
    const float *beta,
    const float *moving_mean,
    const float *moving_variance,
    float epsilon
) {
    int hw = h * w;

    for (int c = 0; c < channels; ++c) {
        float scale = gamma[c] / std::sqrt(moving_variance[c] + epsilon);
        float offset = beta[c] - moving_mean[c] * scale;

        for (int i = 0; i < hw; ++i) {
            map[c][i] = map[c][i] * scale + offset;
        }
    }
}

static void maxpool2x2_map(
    const float in_map[][MAX_PIX],
    int in_h,
    int in_w,
    int channels,
    float out_map[][MAX_PIX]
) {
    int out_h = in_h / 2;
    int out_w = in_w / 2;

    for (int c = 0; c < channels; ++c) {
        for (int y = 0; y < out_h; ++y) {
            for (int x = 0; x < out_w; ++x) {
                float a = in_map[c][(2 * y + 0) * in_w + (2 * x + 0)];
                float b = in_map[c][(2 * y + 0) * in_w + (2 * x + 1)];
                float c0 = in_map[c][(2 * y + 1) * in_w + (2 * x + 0)];
                float d = in_map[c][(2 * y + 1) * in_w + (2 * x + 1)];

                out_map[c][y * out_w + x] =
                    std::max(std::max(a, b), std::max(c0, d));
            }
        }
    }
}

static void global_average_pool_map(
    const float in_map[][MAX_PIX],
    int h,
    int w,
    int channels,
    float out_vec[]
) {
    for (int c = 0; c < channels; ++c) {
        float sum = 0.0f;

        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                sum += in_map[c][y * w + x];
            }
        }

        out_vec[c] = sum / float(h * w);
    }
}

static void dense_compute(
    const float *kernel,
    int in_dim,
    int out_dim,
    const float *biases,
    const float *in_vec,
    float *out_vec
) {
    for (int j = 0; j < out_dim; ++j) {
        float acc = biases ? biases[j] : 0.0f;

        for (int i = 0; i < in_dim; ++i) {
            acc += in_vec[i] * kernel[i * out_dim + j];
        }

        out_vec[j] = acc;
    }
}

static int argmax6(const float output[6]) {
    int best = 0;

    for (int i = 1; i < 6; ++i) {
        if (output[i] > output[best]) {
            best = i;
        }
    }

    return best;
}

extern "C" void audio_inference(const float input[64][64], float output[6]) {
    static float in_map[1][MAX_PIX];

    static float conv1_out[12][MAX_PIX];
    static float pool1_out[12][MAX_PIX];

    static float conv2_out[24][MAX_PIX];
    static float pool2_out[24][MAX_PIX];

    static float conv3_out[48][MAX_PIX];

    static float gap_out[48];
    static float dense_out[6];

    for (int y = 0; y < 64; ++y) {
        for (int x = 0; x < 64; ++x) {
            in_map[0][y * 64 + x] = input[y][x];
        }
    }

    normalize_input_inplace(in_map, 64, 64);

    conv2d_same_map(
        in_map,
        64,
        64,
        1,
        CONV2D_KERNEL,
        CONV2D_KERNEL_K,
        CONV2D_KERNEL_W,
        CONV2D_OUT_CH,
        CONV2D_BIASES,
        conv1_out
    );

    for (int c = 0; c < 12; ++c) {
        for (int i = 0; i < 64 * 64; ++i) {
            conv1_out[c][i] = relu(conv1_out[c][i]);
        }
    }

    batchnorm_inplace(
        conv1_out,
        64,
        64,
        12,
        BATCH_NORMALIZATION_GAMMA,
        BATCH_NORMALIZATION_BETA,
        BATCH_NORMALIZATION_MOVING_MEAN,
        BATCH_NORMALIZATION_MOVING_VARIANCE,
        BATCH_NORMALIZATION_EPSILON
    );

    maxpool2x2_map(conv1_out, 64, 64, 12, pool1_out);

    conv2d_same_map(
        pool1_out,
        32,
        32,
        CONV2D_1_IN_CH,
        CONV2D_1_KERNEL,
        CONV2D_1_KERNEL_K,
        CONV2D_1_KERNEL_W,
        CONV2D_1_OUT_CH,
        CONV2D_1_BIASES,
        conv2_out
    );

    for (int c = 0; c < 24; ++c) {
        for (int i = 0; i < 32 * 32; ++i) {
            conv2_out[c][i] = relu(conv2_out[c][i]);
        }
    }

    batchnorm_inplace(
        conv2_out,
        32,
        32,
        24,
        BATCH_NORMALIZATION_1_GAMMA,
        BATCH_NORMALIZATION_1_BETA,
        BATCH_NORMALIZATION_1_MOVING_MEAN,
        BATCH_NORMALIZATION_1_MOVING_VARIANCE,
        BATCH_NORMALIZATION_1_EPSILON
    );

    maxpool2x2_map(conv2_out, 32, 32, 24, pool2_out);

    conv2d_same_map(
        pool2_out,
        16,
        16,
        CONV2D_2_IN_CH,
        CONV2D_2_KERNEL,
        CONV2D_2_KERNEL_K,
        CONV2D_2_KERNEL_W,
        CONV2D_2_OUT_CH,
        CONV2D_2_BIASES,
        conv3_out
    );

    for (int c = 0; c < 48; ++c) {
        for (int i = 0; i < 16 * 16; ++i) {
            conv3_out[c][i] = relu(conv3_out[c][i]);
        }
    }

    batchnorm_inplace(
        conv3_out,
        16,
        16,
        48,
        BATCH_NORMALIZATION_2_GAMMA,
        BATCH_NORMALIZATION_2_BETA,
        BATCH_NORMALIZATION_2_MOVING_MEAN,
        BATCH_NORMALIZATION_2_MOVING_VARIANCE,
        BATCH_NORMALIZATION_2_EPSILON
    );

    global_average_pool_map(conv3_out, 16, 16, 48, gap_out);

    dense_compute(
        DENSE_KERNEL,
        DENSE_IN,
        DENSE_OUT,
        DENSE_BIASES,
        gap_out,
        dense_out
    );

    for (int j = 0; j < 6; ++j) {
        output[j] = dense_out[j];
    }
}

extern "C" int audio_predict_command_id(const float input[64][64], float output[6]) {
    audio_inference(input, output);
    return argmax6(output);
}

extern "C" void audio_print_prediction(const float input[64][64]) {
    static const char *labels[6] = {
        "go_blue",
        "go_green",
        "go_red",
        "go_yellow",
        "hold",
        "stop"
    };

    float output[6];
    int command_id = audio_predict_command_id(input, output);

    std::printf("Predicted command: %s\n", labels[command_id]);
    std::printf("Logits: ");

    for (int i = 0; i < 6; ++i) {
        std::printf("%f ", output[i]);
    }

    std::printf("\n");
}