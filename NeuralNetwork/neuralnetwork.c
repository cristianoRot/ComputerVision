
// NeuralNetwork.c
#include <stdio.h>
#include <stdlib.h>
#include "neuralnetwork.h" 
#include "Matrix/matrix.h"
#include <math.h>
#include <dirent.h>
#define STB_IMAGE_IMPLEMENTATION
#include "/Users/cristiano/stb_image/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "/Users/cristiano/stb_image/stb_image_write.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "/Users/cristiano/stb_image/stb_image_resize2.h"

double learning_rate = 0.01;

NeuralNetwork* neuralNetwork_create(int* layers, int count_layers) {
    if (layers == NULL || count_layers <= 0) {
        printf("Error: Invalid input parameters.\n");
        return NULL;
    }

    NeuralNetwork* neuralNetwork = (NeuralNetwork*) malloc(sizeof(NeuralNetwork));
    if (neuralNetwork == NULL) {
        printf("Error while allocating memory for the network.\n");
        return NULL;
    }

    Layer* net_layers = (Layer*) malloc(sizeof(Layer) * count_layers);
    if (net_layers == NULL) {
        printf("Error while allocating memory for the layers.\n");
        free(neuralNetwork);  // Free previously allocated memory
        return NULL;
    }

    neuralNetwork->layers = net_layers;
    neuralNetwork->num_layers = count_layers;

    for (int i = 0; i < count_layers; i++) {
        net_layers[i].A = matrix_create(layers[i], 1);
        
        if (net_layers[i].A == NULL) {
            printf("Error while allocating memory for A in layer %d.\n", i);
            // Cleanup allocated memory before returning
            for (int j = 0; j < i; j++) {
                matrix_free(net_layers[j].A);
                if (j > 0) matrix_free(net_layers[j].b);
                if (j < count_layers - 1) matrix_free(net_layers[j].W);
            }
            free(net_layers);
            free(neuralNetwork);
            return NULL;
        }

        if (i > 0) {
            net_layers[i].b = matrix_create(layers[i], 1);
            if (net_layers[i].b == NULL) {
                printf("Error while allocating memory for bias in layer %d.\n", i);
                // Cleanup allocated memory before returning
                for (int j = 0; j <= i; j++) {
                    matrix_free(net_layers[j].A);
                    if (j > 0) matrix_free(net_layers[j].b);
                    if (j < count_layers - 1) matrix_free(net_layers[j].W);
                }
                free(net_layers);
                free(neuralNetwork);
                return NULL;
            }
        }

        if (i < count_layers - 1) {
            net_layers[i].W = matrix_create(layers[i + 1], layers[i]);
            if (net_layers[i].W == NULL) {
                printf("Error while allocating memory for W in layer %d.\n", i);
                // Cleanup allocated memory before returning
                for (int j = 0; j <= i; j++) {
                    matrix_free(net_layers[j].A);
                    if (j > 0) matrix_free(net_layers[j].b);
                    if (j < count_layers - 1) matrix_free(net_layers[j].W);
                }
                free(net_layers);
                free(neuralNetwork);
                return NULL;
            }
        }
    }

    printf("Network created.\n");

    return neuralNetwork;
}

void neuralNetwork_train(NeuralNetwork* network, char* dataset_path) {
    printf("Start training...\n");

    Matrix** dataset = NULL;
    int* label = NULL;

    int total_data = initialize_dataset(&dataset, &label, dataset_path, 19);
    int output_dimension = network->layers[network->num_layers - 1].A->row;
    int epoch = 1;

    while (1) {
        int correct_prediction = 0;

        printf("\n%d° Epoch\n", epoch);
        
        for (int i = 0; i < total_data; i++) {
            Matrix* current_data = dataset[i];

            forward_prop(network, current_data);

            Matrix* onehot_mtx = onehot(label[i], output_dimension);
            int max_output_index = get_max_output_node_index(network);

            if (matrix_get(onehot_mtx, max_output_index, 0) == 1.0)
                correct_prediction++;
            
            back_prop(network, onehot_mtx);

            matrix_free(onehot_mtx);
        }

        double accuracy = (double)correct_prediction / total_data;
        printf("%d° Epoch: %f of accuracy.\n", epoch, accuracy);
    }

    free_dataset(dataset, total_data);
    free(label);
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
    Matrix* dA  = matrix_scalar_product(tmp, 2.0);
    matrix_free(tmp);
    matrix_free(layer->dA);
    layer->dA = dA;

    // • dZ = softmax_backward
    softmax_backward(layer->dZ, layer->A, layer->dA);

    // • dW = dZ · A_prev^T
    tmp = matrix_T(prevLayer->A);
    Matrix *dW  = matrix_product(layer->dZ, tmp);
    matrix_free(tmp);
    matrix_free(layer->dW);
    layer->dW = dW;

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
        Matrix* dA = matrix_product(tmp, nextLayer->dZ);
        matrix_free(tmp);
        matrix_free(layer->dA);
        layer->dA = dA;

        // • dZ(i) = RELU_back(Z(i)) · dA(i)
        RELU_backward(layer->Z, layer->dA, layer->dZ);

        // • dW = dZ · A_prev^T
        tmp = matrix_T(prevLayer->A);
        Matrix *dW  = matrix_product(layer->dZ, tmp);
        matrix_free(tmp);
        matrix_free(layer->dW);
        layer->dW = dW;

        // • db = dZ
        matrix_free(layer->db);
        layer->db = matrix_column_sum(layer->dZ);
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

    for (int i = 0; i < num_classes; i++) {
        matrix_set(result, i, 0, 0.0);
    }

    matrix_set(result, label, 0, 1.0);

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
    return x > 0 ? x : 0;
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
    return pointer > 0 ? 1 : 0;
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

int initialize_dataset(Matrix*** dataset, int** label, char* dataset_path, int num_classes) {
    printf("Initializing dataset...\n");
    int canvas_size = 64;  // Ridimensionato a 64x64
    int total_data = 0;

    // Calcola il numero totale di immagini
    for (int class_index = 0; class_index < num_classes; class_index++) {
        char class_path[256];
        sprintf(class_path, "%s/%d", dataset_path, class_index);

        DIR* dir = opendir(class_path);
        if (!dir) {
            perror("Error opening directory");
            exit(EXIT_FAILURE);
        }

        struct dirent* entry;
        while ((entry = readdir(dir)) != NULL) {
            if (entry->d_type == DT_REG) { // Considera solo file regolari
                total_data++;
            }
        }

        closedir(dir);
    }

    // Allocazione del dataset e della matrice dei label
    *dataset = (Matrix**)malloc(total_data * sizeof(Matrix*));
    *label = malloc(sizeof(int) * total_data);  // Etichette uniche per ogni immagine

    if (*dataset == NULL || *label == NULL) {
        printf("Error allocating memory for dataset or labels.\n");
        exit(EXIT_FAILURE);
    }

    int image_index = 0;

    // Processa ogni immagine
    for (int class_index = 0; class_index < num_classes; class_index++) {
        char class_path[256];
        sprintf(class_path, "%s/%d", dataset_path, class_index);

        DIR* dir = opendir(class_path);
        if (!dir) {
            perror("Error opening directory");
            exit(EXIT_FAILURE);
        }

        struct dirent* entry;
        while ((entry = readdir(dir)) != NULL) {
            if (entry->d_type == DT_REG && entry->d_name[0] != '.') { // Ignora file nascosti
                char image_path[512];
                sprintf(image_path, "%s/%s", class_path, entry->d_name);

                int width, height, channels;
                unsigned char* img_data = stbi_load(image_path, &width, &height, &channels, 1); // Scala di grigi

                if (!img_data) {
                    printf("Error loading image: %s\n", image_path);
                    continue;
                }

                // Ridimensiona l'immagine a 64x64
                unsigned char* resized_img_data = malloc(canvas_size * canvas_size);
                if (resized_img_data == NULL) {
                    printf("Error allocating memory for resized image.\n");
                    stbi_image_free(img_data);
                    exit(EXIT_FAILURE);
                }

                stbir_resize_uint8_linear(img_data, width, height, 0, resized_img_data, canvas_size, canvas_size, 0, 1);
                stbi_image_free(img_data); // Libera l'immagine originale

                // Crea una matrice 64x64
                Matrix* image_matrix = matrix_create(canvas_size * canvas_size, 1);

                for (int i = 0; i < canvas_size * canvas_size; i++) {
                    matrix_set(image_matrix, i, 0, (float)resized_img_data[i] / 255.0);  // Normalizza in [0, 1]
                }

                free(resized_img_data); // Libera la memoria dell'immagine ridimensionata

                // Aggiungi l'immagine al dataset
                (*dataset)[image_index] = image_matrix;

                // Assegna il valore di label corrispondente
                (*label)[image_index] = class_index;

                image_index++;
            }
        }

        closedir(dir);
    }

    printf("Dataset initialized with %d images.\n", total_data);
    return total_data;
}

void free_dataset(Matrix** dataset, int total_images) {
    for (int i = 0; i < total_images; i++) {
        matrix_free(dataset[i]);
    }
    free(dataset);
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

