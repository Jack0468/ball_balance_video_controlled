#include "audio_model.h"
#include <cstdio>

int main() {
    float input[64][64];
    float output[6];

    FILE *file = fopen("test_input_64x64.txt", "r");
    if (!file) {
        printf("Could not open test_input_64x64.txt\n");
        return 1;
    }

    for (int y = 0; y < 64; ++y) {
        for (int x = 0; x < 64; ++x) {
            if (fscanf(file, "%f", &input[y][x]) != 1) {
                printf("Failed to read input value at %d, %d\n", y, x);
                fclose(file);
                return 1;
            }
        }
    }

    fclose(file);

    audio_inference(input, output);

    printf("Logits:\n");
    for (int i = 0; i < 6; ++i) {
        printf("%f ", output[i]);
    }
    printf("\n");

    return 0;
}