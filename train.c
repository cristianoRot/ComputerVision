// train.c

/*
    * This program trains a neural network on a dataset of images.
    * It uses the Accelerate framework for optimized matrix operations.
    * 
    * Compile with:
    * 
    * gcc -DACCELERATE_NEW_LAPACK -o train train.c NeuralNetwork/neuralnetwork.c NeuralNetwork/Matrix/matrix.c NeuralNetwork/DatasetParser/dataset_parser.c -framework Accelerate -lm
    * 
    * To run the program, use:
    * 
    * ./train
    * 
    * Make sure to have the dataset in the "dataset" folder.
    *
    * Note: The Accelerate framework is only available on macOS. For other platforms, you may need to use a different library for matrix operations.
*/

#include <stdio.h>
#include "NeuralNetwork/neuralnetwork.h"
#include "NeuralNetwork/DatasetParser/dataset_parser.h"

int main() {
    Dataset dataset;
    int num_classes = 19;
    int layers[5] = { 4096, 1024, 512, 128, num_classes };

    int epochs = 100;

    printf("Reading dataset...\n");

    int total_data = load_image_folder(&dataset, "dataset", 64, num_classes);

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

    total_data = load_image_folder(&dataset, "dataset", 64, num_classes);

    double accuracy = neuralNetwork_test(
        neuralNetwork,
        &dataset,
        num_classes
    );

    printf("Test accuracy: %.5f%%\n", accuracy * 100.0);
}