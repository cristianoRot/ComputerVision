// main.c
#include <stdio.h>
#include "NeuralNetwork/neuralnetwork.h"
#include "NeuralNetwork/Matrix/matrix.h" 

int main() {
    int layers[5] = { 1024, 512, 128, 256, 19};
    NeuralNetwork* neuralNetwork = neuralNetwork_create(layers, 5);

    if (neuralNetwork == NULL) {
        printf("Error while creating network.\n");
        return 1;
    }

    neuralNetwork_train(neuralNetwork, "/Users/cristiano/Desktop/ComputerVision/dataset", 10);
}