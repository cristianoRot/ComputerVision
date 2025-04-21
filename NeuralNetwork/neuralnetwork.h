#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "Matrix/matrix.h"
#include <stdlib.h>

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

void neuralNetwork_train(NeuralNetwork* network, char* dataset_path);

int neuralNetwork_predict(NeuralNetwork* network, Matrix* input);

void adjust_learning_rate(int epoch);

void back_prop(NeuralNetwork* network, Matrix* one_hot);

void forward_prop(NeuralNetwork* network, Matrix* input);

Matrix* onehot(int output_dimension, int index);

void softmax(const Matrix *in, Matrix *out);

void softmax_backward(Matrix* dZ, Matrix* A, Matrix* dA);

double RELU(double x);

void RELU_matrix(Matrix* in, Matrix* out);

char RELU_der(double pointer);

void RELU_backward(const Matrix* Z, Matrix* dA, Matrix* dZ);

int get_max_output_node_index(NeuralNetwork* network);

int initialize_dataset(Matrix*** dataset, int** label, char* dataset_path, int num_classes);

void free_dataset(Matrix** dataset, int total_data);

void save_model(NeuralNetwork* network, char* filename);

NeuralNetwork* load_model(const char* filename);

#endif