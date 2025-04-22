// NeuralNetwork.c
#include <stdio.h>
#include <stdlib.h>
#include "neuralnetwork.h" 
#include "Matrix/matrix.h"
#include <math.h>
#include <dirent.h>
#include <Accelerate/Accelerate.h>
#define STB_IMAGE_IMPLEMENTATION
#include "/Users/cristiano/stb_image/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "/Users/cristiano/stb_image/stb_image_write.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "/Users/cristiano/stb_image/stb_image_resize2.h"

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

void neuralNetwork_train(NeuralNetwork* network, const char* dataset_path, int epochs) {
    printf("Start training...\n");

    Dataset dataset;
    int total_data = initialize_dataset(&dataset, dataset_path, dataset_path, 19);

    int num_layers = network->num_layers;

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        printf("=== Epoch %d ===\n", epoch);

        int correct_prediction = 0;
        
        for (int i = 0; i < total_data; i++) {
            Matrix* x = dataset.X[i];
            Matrix* y_true = dataset.Y[i];

            forward_prop(network, x);

            int max_output_index = get_max_output_node_index(network);

            if (matrix_get(y_true, max_output_index, 0) == 1.0) {
                correct_prediction++;
            }
            
            back_prop(network, y_true);
        }

        double accuracy = (double)correct_prediction / total_data;
        printf("%d° Epoch: %f of accuracy.\n", epoch, accuracy);

        adjust_learning_rate(epoch);
    }

    free_dataset(&dataset);
}

int neuralNetwork_predict(NeuralNetwork* network, Matrix* input) {
    forward_prop(network, input);
    return get_max_output_node_index(network);
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

    // • dA = 2*(A - y_true)
    Matrix* tmp = matrix_sub(layer->A, y_true);
    matrix_free(layer->dA);
    layer->dA  = matrix_scalar_product(tmp, 2.0);
    matrix_free(tmp);

    // • dZ = softmax_backward
    softmax_backward(layer->dZ, layer->A, layer->dA);

    // • dW = dZ · A_prev^T
    tmp = matrix_T(prevLayer->A);
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

int initialize_dataset(Dataset* dataset, const char* dataset_path, int canvas_size, int num_classes) {
    printf("Initializing dataset...\n");
    int total_images = 0;
    // First pass: count all images
    for (int class_index = 0; class_index < num_classes; ++class_index) {
        char class_path[256];
        sprintf(class_path, "%s/%d", dataset_path, class_index);
        DIR* dir = opendir(class_path);
        if (!dir) {
            perror("Error opening directory");
            exit(EXIT_FAILURE);
        }
        struct dirent* entry;
        while ((entry = readdir(dir)) != NULL) {
            if (entry->d_type == DT_REG && entry->d_name[0] != '.') {
                total_images++;
            }
        }
        closedir(dir);
    }
    dataset->N = total_images;
    dataset->X = (Matrix**)malloc(dataset->N * sizeof(Matrix*));
    dataset->Y = (Matrix**)malloc(dataset->N * sizeof(Matrix*));
    if (!dataset->X || !dataset->Y) {
        printf("Error allocating memory for dataset.\n");
        exit(EXIT_FAILURE);
    }
    int img_idx = 0;
    for (int class_index = 0; class_index < num_classes; ++class_index) {
        char class_path[256];
        sprintf(class_path, "%s/%d", dataset_path, class_index);
        DIR* dir = opendir(class_path);
        if (!dir) {
            perror("Error opening directory");
            exit(EXIT_FAILURE);
        }
        struct dirent* entry;
        while ((entry = readdir(dir)) != NULL) {
            if (entry->d_type == DT_REG && entry->d_name[0] != '.') {
                char image_path[512];
                sprintf(image_path, "%s/%s", class_path, entry->d_name);
                int width, height, channels;
                unsigned char* img_data = stbi_load(image_path, &width, &height, &channels, 1);
                if (!img_data) {
                    printf("Error loading image: %s\n", image_path);
                    continue;
                }
                unsigned char* resized_img_data = malloc(canvas_size * canvas_size);
                if (!resized_img_data) {
                    printf("Error allocating memory for resized image.\n");
                    stbi_image_free(img_data);
                    exit(EXIT_FAILURE);
                }
                stbir_resize_uint8_linear(
                    img_data, width, height, 0,
                    resized_img_data, canvas_size, canvas_size, 0, 1
                );
                stbi_image_free(img_data);
                Matrix* image_matrix = matrix_create(canvas_size * canvas_size, 1);
                for (int i = 0; i < canvas_size * canvas_size; ++i) {
                    matrix_set(image_matrix, i, 0, (float)resized_img_data[i] / 255.0f);
                }
                free(resized_img_data);
                dataset->X[img_idx] = image_matrix;
                dataset->Y[img_idx] = onehot(class_index, num_classes);
                img_idx++;
            }
        }
        closedir(dir);
    }
    printf("Dataset initialized with %d images.\n", dataset->N);
    return dataset->N;
}

void free_dataset(Dataset* dataset) {
    // Free each feature and label matrix
    for (int i = 0; i < dataset->N; ++i) {
        matrix_free(dataset->X[i]);
        matrix_free(dataset->Y[i]);
    }
    // Free the arrays of pointers
    free(dataset->X);
    free(dataset->Y);
    // Reset fields
    dataset->X = NULL;
    dataset->Y = NULL;
    dataset->N = 0;
}

void save_model(NeuralNetwork* network, char* filename) {
    FILE* file = fopen(filename, "wb");
    if (file == NULL) {
        perror("Error opening file for saving model");
        return;
    }

    fwrite(&(network->num_layers), sizeof(int), 1, file);

    for (int i = 0; i < network->num_layers; i++) {
        Layer layer = network->layers[i];

        int rows = layer.A->row;
        int cols = layer.A->col;
        fwrite(&rows, sizeof(int), 1, file);
        fwrite(&cols, sizeof(int), 1, file);

        if (layer.W != NULL) {
            fwrite(&(layer.W->row), sizeof(int), 1, file);
            fwrite(&(layer.W->col), sizeof(int), 1, file);
            for (int r = 0; r < layer.W->row; r++) {
                fwrite(layer.W->data[r], sizeof(double), layer.W->col, file);
            }
        }

        if (layer.b != NULL) {
            fwrite(&(layer.b->row), sizeof(int), 1, file);
            fwrite(&(layer.b->col), sizeof(int), 1, file);
            fwrite(layer.b->data[0], sizeof(double), layer.b->row * layer.b->col, file);
        }
    }

    fclose(file);
}

NeuralNetwork* load_model(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (file == NULL) {
        perror("Error opening file for loading model.\n");
        return NULL;
    }

    NeuralNetwork* network = malloc(sizeof(NeuralNetwork));
    fread(&(network->num_layers), sizeof(int), 1, file);

    network->layers = malloc(sizeof(Layer) * network->num_layers);

    for (int i = 0; i < network->num_layers; i++) {
        Layer* layer = &(network->layers[i]);

        int rows, cols;
        fread(&rows, sizeof(int), 1, file);
        fread(&cols, sizeof(int), 1, file);
        layer->A = matrix_create(rows, cols);

        if (i < network->num_layers - 1) {
            fread(&rows, sizeof(int), 1, file);
            fread(&cols, sizeof(int), 1, file);
            layer->W = matrix_create(rows, cols);
            for (int r = 0; r < rows; r++) {
                fread(layer->W->data[r], sizeof(double), cols, file);
            }
        }

        if (i > 0) {
            fread(&rows, sizeof(int), 1, file);
            fread(&cols, sizeof(int), 1, file);
            layer->b = matrix_create(rows, cols);
            fread(layer->b->data[0], sizeof(double), rows * cols, file);
        }
    }

    fclose(file);
    return network;
}

