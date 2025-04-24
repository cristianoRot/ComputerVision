// main.c
#include <stdio.h>
#include "NeuralNetwork/neuralnetwork.h"
#include "NeuralNetwork/DatasetParser/dataset_parser.h"

int main() {

    Dataset dataset;
    int num_classes = 2;
    int layers[3] = { 2, 2, num_classes};

    int epochs = 10000;

    printf("Reading dataset...\n");

    int total_data = load_csv(&dataset, "/Users/cristiano/Desktop/XOR_Dataset.csv", 2, num_classes);

    NeuralNetwork* neuralNetwork = neuralNetwork_create(layers, sizeof(layers) / sizeof(int));

    if (neuralNetwork == NULL) {
        printf("Error while creating network.\n");
        return 1;
    }

    neuralNetwork_train(
        neuralNetwork, 
        &dataset,
        "/Users/cristiano/Desktop/XOR_model",
        num_classes,
        epochs
    );

    free_dataset(&dataset);

    total_data = load_csv(&dataset, "/Users/cristiano/Desktop/XOR_Dataset.csv", 2, num_classes);

    double accuracy = neuralNetwork_test(
        neuralNetwork,
        &dataset,
        num_classes
    );

    printf("Test accuracy: %.5f%%\n", accuracy * 100.0);
}