#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "stb_image/stb_image.h"
#include "stb_image/stb_image_write.h"
#include "stb_image/stb_image_resize2.h"

#include "NeuralNetwork/neuralnetwork.h"
#include "NeuralNetwork/Matrix/matrix.h"


Matrix* load_image_as_matrix(const char* filepath, int expected_input_size);

int main() {
    const char* model_path = "model";
    const char* test_image_path = "testImage.png";

    int num_classes = 26;
    int layers[5] = { 784, 64, 128, 32, num_classes };
    int num_layers = sizeof(layers) / sizeof(layers[0]);

    NeuralNetwork* network = NULL;
    Matrix* test_input_matrix = NULL;

    network = neuralNetwork_create(layers, num_layers);
    if (!network) {
        fprintf(stderr, "Error: Unable to create neural network.\n");
        return 1;
    }

    load_model(network, model_path);

    test_input_matrix = load_image_as_matrix(test_image_path, 28);

    if (!test_input_matrix) {
        fprintf(stderr, "Error: Unable to load image as matrix.\n");
        return 1;
    }

    int predicted_class_index = neuralNetwork_predict(network, test_input_matrix);

    printf("Predicted Class Index: %c\n", predicted_class_index + 'A');
}


Matrix* load_image_as_matrix(const char* filepath, int canvas_size) {
    int w, h, channels;
    unsigned char* data = stbi_load(filepath, &w, &h, &channels, 0);

    if (!data) {
        return NULL;
    }

    unsigned char* gray_orig = malloc(w * h);

    if (!gray_orig) {
        stbi_image_free(data);
        return NULL;
    }

    for (int i = 0; i < w * h; ++i) {
         int sum = 0;
         for (int ch = 0; ch < channels; ++ch) {
             sum += data[i * channels + ch];
         }
         gray_orig[i] = (unsigned char)(sum / channels);
    }

    stbi_image_free(data);

    unsigned char* resized = malloc(canvas_size * canvas_size);
    stbir_resize_uint8_linear(gray_orig, w, h, 0,
                              resized, canvas_size, canvas_size, 0, 1);
    free(gray_orig);

    int final_num_features = canvas_size * canvas_size;

    Matrix* img_matrix = matrix_create(canvas_size * canvas_size, 1);

    for (int i = 0; i < final_num_features; ++i) {
        img_matrix->data[i][0] = (double)resized[i] / 255.0;
    }

    free(resized);

    return img_matrix;
}