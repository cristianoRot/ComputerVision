// main.c
#include <stdio.h>
#include "NeuralNetwork/neuralnetwork.h"
#include "DatasetParser/dataset_parser.h"

int main() {

    Dataset* dataset;

    dataset_parse(dataset);

    int layers[5] = { 784, 64, 128, 32, 19 };
    NeuralNetwork* neuralNetwork = neuralNetwork_create(layers, 5);

    if (neuralNetwork == NULL) {
        printf("Error while creating network.\n");
        return 1;
    }

    neuralNetwork_train(
        neuralNetwork, 
        dataset,
        "/Users/cristiano/Desktop/ComputerVision/model",
        19,
        10
    );
}