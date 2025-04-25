#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "DatasetParser/dataset_parser.h"

typedef struct {
    unsigned char r;
    unsigned char g;
    unsigned char b;
} Color;

typedef struct {
    Matrix* A;
    Matrix* b;
    Matrix* W;
    Matrix* Z;

    Matrix* dA;
    Matrix* db;
    Matrix* dW;
    Matrix* dZ;

} Layer;

typedef struct {
    int num_layers;
    Layer* layers;
} NeuralNetwork;


NeuralNetwork* neuralNetwork_create(int* layer_dims, int count_layers);

void neuralNetwork_train(NeuralNetwork* network, Dataset* dataset, const char* model_path_, int num_classes, int epochs);

double neuralNetwork_test(NeuralNetwork* network, Dataset* dataset, int num_classes);

int neuralNetwork_predict(NeuralNetwork* network, Matrix* input);

int* FisherYates_shuffle(int size);

void adjust_learning_rate(int epoch);

void back_prop(NeuralNetwork* network, Matrix* one_hot);

void forward_prop(NeuralNetwork* network, Matrix* input);

Matrix* onehot(int label, int num_classes);

void softmax(const Matrix *in, Matrix *out);

double RELU(double x);

void RELU_matrix(const Matrix* matrixIn, Matrix* matrixOut);

double RELU_der(double pointer);

void RELU_backward(const Matrix* Z, const Matrix* dA, Matrix* dZ);

int get_max_output_node_index(Matrix* output_layer);

void save_model(NeuralNetwork* network, const char* filename);

void load_model(NeuralNetwork* network, const char* filename);

#endif