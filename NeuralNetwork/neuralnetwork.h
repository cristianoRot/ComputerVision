// NeuralNetwork.h
#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "Matrix/matrix.h"

typedef struct {
    unsigned char r;
    unsigned char g;
    unsigned char b;
} Color;

typedef struct{
    Matrix* nodes;
    Matrix* bias;
    Matrix* weights;
} Layers;

typedef struct {
    int num_layers;
    Layers* layers;
} NeuralNetwork;

NeuralNetwork* neuralNetwork_create(int* layers, int count_layers);

void neuralNetwork_train(NeuralNetwork* network, char* dataset_path);

int neuralNetwork_predict(NeuralNetwork* network, Matrix* input);

void adjust_learning_rate(int epoch);

void back_prop(NeuralNetwork* network, Matrix* one_hot);

void forward_prop(NeuralNetwork* network, Matrix* input);

Matrix* onehot(int output_dimension, int index);

void softmax(Matrix* matrix);

void RELU(double* pointer);

void RELU_matrix(Matrix* pointer);

char RELU_der(double pointer);

Matrix* RELU_der_matrix(Matrix* matrix);

int get_max_output_node_index(NeuralNetwork* network);

int initialize_dataset(Matrix*** dataset, int** label, char* dataset_path, int num_classes);

void free_dataset(Matrix** dataset, int total_data);

void save_model(NeuralNetwork* network, char* filename);

NeuralNetwork* load_model(const char* filename);

#endif