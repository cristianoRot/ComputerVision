// NeuralNetwork.c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include "neuralnetwork.h" 
#include "Matrix/matrix.h"
#include <math.h>
#include <dirent.h>
#include <Accelerate/Accelerate.h>

const char* model_path;
double learning_rate = 0.01;

NeuralNetwork* neuralNetwork_create(int* layer_dims, int num_layers) {
    NeuralNetwork* net = malloc(sizeof(NeuralNetwork));
    if (!net) return NULL;
    net->num_layers = num_layers;
    // Allocate and zero-initialize the layers array
    net->layers = calloc(num_layers, sizeof(Layer));
    if (!net->layers) {
        free(net);
        return NULL;
    }

    for (int i = 0; i < num_layers; ++i) {
        Layer* L = &net->layers[i];
        int rows = layer_dims[i];

        // Always allocate A, Z, dA, dZ as [rows × 1]
        if (i > 0) {
            L->A  = matrix_create(rows, 1); // Will be filled during forward pass
            L->Z  = matrix_create(rows, 1);
            L->dA = matrix_create(rows, 1);
            L->dZ = matrix_create(rows, 1);
        } else {
            L->A = NULL; // Will be assigned the input matrix
            L->Z = NULL;
            L->dA = NULL;
            L->dZ = NULL;
        }

        if (i > 0) {
            int prev = layer_dims[i-1];
            // Weights and their gradient: [rows × prev]
            L->W  = matrix_random(rows, prev);
            L->dW = matrix_create(rows, prev);
            // Bias and its gradient: [rows × 1]
            L->b  = matrix_create(rows, 1);
            L->db = matrix_create(rows, 1);
        } else {
            // Input layer has no params
            L->W = L->dW = NULL;
            L->b = L->db = NULL;
        }
    }

    return net;
}

void neuralNetwork_train(NeuralNetwork* network, Dataset* dataset, const char* model_path_, int num_classes, int epochs) {
    printf("Start training...\n");

    model_path = model_path_;

    // Load model if exists
    if (access(model_path, F_OK) == 0) {
        load_model(network, model_path);
        printf("Loaded existing model from %s\n", model_path);
    }

    int num_layers = network->num_layers;
    double lastAccuracy = 0.0;

    // Seed the random number generator once
    srand((unsigned)time(NULL));

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        printf("=== Epoch %d ===\n", epoch);

        int correct_prediction = 0;

        int* indices = FisherYates_shuffle(dataset->N);

        for (int k = 0; k < dataset->N; ++k) {
            int i = indices[k];
            Matrix* x = dataset->X[i];
            Matrix* y_true = dataset->Y[i];

            forward_prop(network, x);

            int max_output_index = get_max_output_node_index(network);

            if (matrix_get(y_true, max_output_index, 0) == 1.0) {
                correct_prediction++;
            }
            
            back_prop(network, y_true);
        }
        free(indices);

        double accuracy = (double)correct_prediction / dataset->N;

        if (accuracy > lastAccuracy) {
            save_model(network, model_path);
            lastAccuracy = accuracy;
        }

        printf("Epoch %d accuracy: %.2f%%\n", epoch, accuracy * 100.0);

        adjust_learning_rate(epoch);
    }

    free_dataset(&dataset);
}

int neuralNetwork_predict(NeuralNetwork* network, Matrix* input) {
    forward_prop(network, input);
    return get_max_output_node_index(network);
}

int* FisherYates_shuffle(int size) {
    int *indices = malloc(size * sizeof(int));

    for (int i = 0; i < size; ++i) {
        indices[i] = i;
    }

    for (int i = size - 1; i > 0; --i) {
        int j = rand() % (i + 1);
        int tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
    }

    return indices;
}

void adjust_learning_rate(int epoch) {
    if (epoch % 10 == 0 && epoch != 0) {
        learning_rate *= 0.9;
    }
}

void back_prop(NeuralNetwork* network, Matrix* y_true) {
    int last = network->num_layers - 1;
    Layer* layer = &network->layers[last];
    Layer* prevLayer = &network->layers[last - 1];

    // --- OUTPUT LAYER ---

    // Cross‑entropy loss backward (softmax + CE): dZ = A - y_true
    matrix_free(layer->dZ);
    layer->dZ = matrix_sub(layer->A, y_true);

    // • dW = dZ · A_prev^T
    Matrix* tmp = matrix_T(prevLayer->A);
    matrix_free(layer->dW);
    layer->dW  = matrix_product(layer->dZ, tmp);
    matrix_free(tmp);

    // • db = dZ
    matrix_free(layer->db);
    layer->db = matrix_column_sum(layer->dZ);

    // --- HIDDEN LAYERS ---

    for (int i = last - 1; i > 0; i--)
    {
        layer = &network->layers[i];
        prevLayer = &network->layers[i - 1];
        Layer* nextLayer = &network->layers[i + 1];

        // • dA(i) = W_T(i + 1) · dZ(i + 1)
        tmp = matrix_T(nextLayer->W);
        matrix_free(layer->dA);
        layer->dA = matrix_product(tmp, nextLayer->dZ);
        matrix_free(tmp);

        // • dZ(i) = RELU_back(Z(i)) · dA(i)
        RELU_backward(layer->Z, layer->dA, layer->dZ);

        // • dW = dZ · A_prev^T
        tmp = matrix_T(prevLayer->A);
        matrix_free(layer->dW);
        layer->dW  = matrix_product(layer->dZ, tmp);
        matrix_free(tmp);

        // • db = dZ
        matrix_free(layer->db);
        layer->db = matrix_column_sum(layer->dZ);
    }

    for (int l = 1; l < network->num_layers; ++l) {
        Layer* layer = &network->layers[l];
    
        for (int i = 0; i < layer->W->row; ++i) {
            for (int j = 0; j < layer->W->col; ++j) {
                layer->W->data[i][j] -= learning_rate * layer->dW->data[i][j];
            }
        }
    
        for (int i = 0; i < layer->b->row; ++i) {
            layer->b->data[i][0] -= learning_rate * layer->db->data[i][0];
        }
    }
}

void forward_prop(NeuralNetwork* network, Matrix* input) {
    int L = network->num_layers;

    network->layers[0].A = input;

    for (int i = 1; i < L; ++i) {
        Layer *layer = &network->layers[i];
        Layer *prev_layer = &network->layers[i-1];

        // Z = W · A_prev + b
        Matrix *prod = matrix_product(layer->W, prev_layer->A);
        matrix_free(layer->Z);
        layer->Z = matrix_sum(prod, layer->b);
        matrix_free(prod);

        if (i < L-1) {
            RELU_matrix(layer->Z, layer->A);
        } else {
            softmax(layer->Z, layer->A);
        }
    }
}

Matrix* onehot(int label, int num_classes) {
    Matrix* result = matrix_create(num_classes, 1);
    if (!result) {
        perror("Error creating one-hot matrix");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < num_classes; i++) {
        matrix_set(result, i, 0, 0.0);
    }

    if (label >= 0 && label < num_classes) {
        matrix_set(result, label, 0, 1.0);
    } else {
        fprintf(stderr, "Warning: Label %d is out of bounds for %d classes.\n", label, num_classes);
    }

    return result;
}

void softmax(const Matrix *in, Matrix *out) {
    int R = in->row;
    int C = in->col;

    for (int j = 0; j < C; ++j) {
        double max_val = in->data[0][j];
        for (int i = 1; i < R; ++i) {
            double v = in->data[i][j];
            if (v > max_val) {
                max_val = v;
            }
        }

        double sum_exp = 0.0;
        for (int i = 0; i < R; ++i) {
            double e = exp(in->data[i][j] - max_val);
            out->data[i][j] = e;
            sum_exp += e;
        }

        if (sum_exp == 0.0) {
            double uniform = 1.0 / R;
            for (int i = 0; i < R; ++i) {
                out->data[i][j] = uniform;
            }
        } else {
            for (int i = 0; i < R; ++i) {
                out->data[i][j] /= sum_exp;
            }
        }
    }
}

void softmax_backward(Matrix* dZ, Matrix* A, Matrix* dA) {
    int C = A->row;
    int M = A->col;

    for (int j = 0; j < M; ++j) {
        for (int i = 0; i < C; ++i) {
            double sum = 0.0;
            for (int k = 0; k < C; ++k) {
                double ai = A->data[i][j];
                double ak = A->data[k][j];
                double jac = (i == k)
                             ? ai * (1.0 - ai)
                             : -ai * ak;
                sum += jac * dA->data[k][j];
            }
            dZ->data[i][j] = sum;
        }
    }
}

double RELU(double x) {
    return x > 0 ? x : 0.0;
}

void RELU_matrix(Matrix* matrixIn, Matrix* matrixOut) {
    int row_count = matrixIn->row;
    int col_count = matrixIn->col;

    for (int r = 0; r < row_count; r++) {
        for (int c = 0; c < col_count; c++) {
            matrixOut->data[r][c] = RELU(matrixIn->data[r][c]);
        }
    }
}

char RELU_der(double pointer) {
    return pointer > 0 ? 1.0 : 0.0;
}

void RELU_backward(const Matrix* Z, Matrix* dA, Matrix* dZ) {
    int rows = Z->row;
    int cols = Z->col;

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) 
        {
            char act_der = RELU_der(Z->data[r][c]);
            dZ->data[r][c] = dA->data[r][c] * act_der;
        }
    }
}

int get_max_output_node_index(NeuralNetwork* network) {
    Matrix* output_layer = network->layers[network->num_layers - 1].A;

    double max = matrix_get(output_layer, 0, 0);
    int index = 0;

    for (int i = 1; i < output_layer->row; i++) {
        if (matrix_get(output_layer, i, 0) > max) {
            max = matrix_get(output_layer, i, 0);
            index = i;
        }
    }

    return index;
}

void save_model(NeuralNetwork* network, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        perror("Error opening file for saving model");
        return;
    }

    // Write magic number
    const char magic[4] = {'N','N','0','1'};
    if (fwrite(magic, 1, 4, file) != 4) {
        perror("Error writing model magic");
        fclose(file);
        return;
    }

    // Write number of layers
    int num_layers = network->num_layers;
    if (fwrite(&num_layers, sizeof(int), 1, file) != 1) {
        perror("Error writing num_layers");
        fclose(file);
        return;
    }

    // Write array of layer dimensions
    int* dims = malloc(num_layers * sizeof(int));
    if (!dims) {
        perror("Error allocating dims array");
        fclose(file);
        return;
    }
    // Input layer dimension: get from first layer's A or W
    if (network->layers[0].A) {
        dims[0] = network->layers[0].A->row;
    } else if (network->layers[1].W) {
        dims[0] = network->layers[1].W->col;
    } else {
        dims[0] = 0;
    }
    for (int i = 1; i < num_layers; ++i) {
        if (network->layers[i].W)
            dims[i] = network->layers[i].W->row;
        else if (network->layers[i].A)
            dims[i] = network->layers[i].A->row;
        else
            dims[i] = 0;
    }
    if (fwrite(dims, sizeof(int), num_layers, file) != (size_t)num_layers) {
        perror("Error writing layer dims");
        free(dims);
        fclose(file);
        return;
    }

    // For each layer i=1..num_layers-1, write W and b
    for (int i = 1; i < num_layers; ++i) {
        Layer* L = &network->layers[i];
        // Write weights matrix (rows*cols doubles, row-major)
        int rows = L->W->row, cols = L->W->col;
        for (int r = 0; r < rows; ++r) {
            if (fwrite(L->W->data[r], sizeof(double), cols, file) != (size_t)cols) {
                perror("Error writing weight data");
                free(dims);
                fclose(file);
                return;
            }
        }
        // Write bias vector (rows doubles)
        if (fwrite(L->b->data[0], sizeof(double), rows, file) != (size_t)rows) {
            perror("Error writing bias data");
            free(dims);
            fclose(file);
            return;
        }
    }
    free(dims);
    fclose(file);
}

void load_model(NeuralNetwork* network, const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        perror("Error opening file for loading model");
        return;
    }

    // Read and check magic
    char magic[4];
    if (fread(magic, 1, 4, file) != 4 || magic[0] != 'N' || magic[1] != 'N' || magic[2] != '0' || magic[3] != '1') {
        fprintf(stderr, "Model file has invalid magic header\n");
        fclose(file);
        return;
    }

    // Read number of layers
    int num_layers;
    if (fread(&num_layers, sizeof(int), 1, file) != 1) {
        perror("Error reading num_layers");
        fclose(file);
        return;
    }

    // Read dims array
    int* dims = malloc(num_layers * sizeof(int));
    if (!dims) {
        perror("Error allocating dims array");
        fclose(file);
        return;
    }
    if (fread(dims, sizeof(int), num_layers, file) != (size_t)num_layers) {
        perror("Error reading layer dims");
        free(dims);
        fclose(file);
        return;
    }

    // Free existing layers
    if (network->layers) {
        for (int i = 0; i < network->num_layers; ++i) {
            Layer* L = &network->layers[i];
            if (L->A) matrix_free(L->A);
            if (L->Z) matrix_free(L->Z);
            if (L->dA) matrix_free(L->dA);
            if (L->dZ) matrix_free(L->dZ);
            if (L->W) matrix_free(L->W);
            if (L->dW) matrix_free(L->dW);
            if (L->b) matrix_free(L->b);
            if (L->db) matrix_free(L->db);
        }
        free(network->layers);
    }

    network->num_layers = num_layers;
    network->layers = calloc(num_layers, sizeof(Layer));
    if (!network->layers) {
        perror("Error allocating layers array");
        free(dims);
        fclose(file);
        return;
    }

    for (int i = 0; i < num_layers; ++i) {
        Layer* L = &network->layers[i];
        int rows = dims[i];
        if (i > 0) {
            L->A  = matrix_create(rows, 1);
            L->Z  = matrix_create(rows, 1);
            L->dA = matrix_create(rows, 1);
            L->dZ = matrix_create(rows, 1);
        } else {
            L->A = NULL;
            L->Z = NULL;
            L->dA = NULL;
            L->dZ = NULL;
        }
        if (i > 0) {
            int prev = dims[i-1];
            L->W  = matrix_create(rows, prev);
            L->dW = matrix_create(rows, prev);
            L->b  = matrix_create(rows, 1);
            L->db = matrix_create(rows, 1);
        } else {
            L->W = L->dW = NULL;
            L->b = L->db = NULL;
        }
    }

    // For each layer i=1..num_layers-1, read W and b
    for (int i = 1; i < num_layers; ++i) {
        Layer* L = &network->layers[i];
        int rows = L->W->row, cols = L->W->col;
        for (int r = 0; r < rows; ++r) {
            if (fread(L->W->data[r], sizeof(double), cols, file) != (size_t)cols) {
                perror("Error reading weight data");
                free(dims);
                fclose(file);
                return;
            }
        }
        if (fread(L->b->data[0], sizeof(double), rows, file) != (size_t)rows) {
            perror("Error reading bias data");
            free(dims);
            fclose(file);
            return;
        }
    }
    free(dims);
    fclose(file);
}

