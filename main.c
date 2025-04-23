// main.c
#include <stdio.h>
#include "NeuralNetwork/neuralnetwork.h"
#include "NeuralNetwork/DatasetParser/dataset_parser.h"

int main() {

    Dataset dataset;

    printf("Reading dataset...\n");

    int total_data = load_image_folder(&dataset, "/Users/cristiano/Desktop/ComputerVision/dataset", 32, 19);

    if (total_data == -1) {
        printf("Error loading dataset.\n");
        return 1;
    }
    else {
        printf("Dataset loaded with %d images.\n", total_data);
    }

    int layers[5] = { 1024, 512, 256, 128, 19 };
    NeuralNetwork* neuralNetwork = neuralNetwork_create(layers, 5);

    if (neuralNetwork == NULL) {
        printf("Error while creating network.\n");
        return 1;
    }

    neuralNetwork_train(
        neuralNetwork, 
        &dataset,
        "/Users/cristiano/Desktop/ComputerVision/model",
        19,
        200
    );
}