// train.c
#include <stdio.h>
#include "NeuralNetwork/neuralnetwork.h"
#include "NeuralNetwork/DatasetParser/dataset_parser.h"

int main() {

    Dataset dataset;
    int num_classes = 26;
    int layers[5] = { 784, 64, 128, 32, num_classes };

    int epochs = 10;

    printf("Reading dataset...\n");

    int total_data = load_csv(&dataset, "/Users/cristiano/Desktop/DigitRecognizer/digital_letters.csv", 784, num_classes);

    NeuralNetwork* neuralNetwork = neuralNetwork_create(layers, sizeof(layers) / sizeof(int));

    if (neuralNetwork == NULL) {
        printf("Error while creating network.\n");
        return 1;
    }

    neuralNetwork_train(
        neuralNetwork, 
        &dataset,
        "model",
        num_classes,
        epochs
    );

    free_dataset(&dataset);

    total_data = load_csv(&dataset, "/Users/cristiano/Desktop/DigitRecognizer/digital_letters.csv", 28, num_classes);

    double accuracy = neuralNetwork_test(
        neuralNetwork,
        &dataset,
        num_classes
    );

    printf("Test accuracy: %.5f%%\n", accuracy * 100.0);
}