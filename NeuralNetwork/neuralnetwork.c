
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

    Layers* net_layers = (Layers*) malloc(sizeof(Layers) * count_layers);
    if (net_layers == NULL) {
        printf("Error while allocating memory for the layers.\n");
        free(neuralNetwork);  // Free previously allocated memory
        return NULL;
    }

    neuralNetwork->layers = net_layers;
    neuralNetwork->num_layers = count_layers;

    for (int i = 0; i < count_layers; i++) {
        net_layers[i].nodes = matrix_create(layers[i], 1);
        
        if (net_layers[i].nodes == NULL) {
            printf("Error while allocating memory for nodes in layer %d.\n", i);
            // Cleanup allocated memory before returning
            for (int j = 0; j < i; j++) {
                free_matrix(net_layers[j].nodes);
                if (j > 0) free_matrix(net_layers[j].bias);
                if (j < count_layers - 1) free_matrix(net_layers[j].weights);
            }
            free(net_layers);
            free(neuralNetwork);
            return NULL;
        }

        if (i > 0) {
            net_layers[i].bias = matrix_create(layers[i], 1);
            if (net_layers[i].bias == NULL) {
                printf("Error while allocating memory for bias in layer %d.\n", i);
                // Cleanup allocated memory before returning
                for (int j = 0; j <= i; j++) {
                    free_matrix(net_layers[j].nodes);
                    if (j > 0) free_matrix(net_layers[j].bias);
                    if (j < count_layers - 1) free_matrix(net_layers[j].weights);
                }
                free(net_layers);
                free(neuralNetwork);
                return NULL;
            }
        }

        if (i < count_layers - 1) {
            net_layers[i].weights = matrix_create(layers[i + 1], layers[i]);
            if (net_layers[i].weights == NULL) {
                printf("Error while allocating memory for weights in layer %d.\n", i);
                // Cleanup allocated memory before returning
                for (int j = 0; j <= i; j++) {
                    free_matrix(net_layers[j].nodes);
                    if (j > 0) free_matrix(net_layers[j].bias);
                    if (j < count_layers - 1) free_matrix(net_layers[j].weights);
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
    int output_dimension = network->layers[network->num_layers - 1].nodes->row;
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

            free_matrix(onehot_mtx);
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

void back_prop(NeuralNetwork* network, Matrix* one_hot) {
    if (network == NULL || one_hot == NULL) {
        printf("ERROR\n");
        return; 
    }

    int last_index = network->num_layers - 1; 
    Matrix* loss = matrix_sub(network->layers[last_index].nodes, one_hot);

    if (loss == NULL) {
        printf("Error in loss\n");
        return;
    }    

    for (int layer = last_index; layer > 0; layer--) {
        printf("Current layer %d\n", layer);

        Matrix* prev_layer_T = matrix_T(network->layers[layer - 1].nodes);

        if (prev_layer_T == NULL) {
            printf("Error in prev_layer_T\n");
            return;
        }  

        Matrix* grad_W_ = matrix_product(loss, prev_layer_T);
        if (grad_W_ == NULL) {
            printf("Error in grad_W_\n");
            return;
        }  

        Matrix* grad_W = matrix_scalar_product(grad_W_, learning_rate);
        if (grad_W == NULL) {
            printf("Error in grad_W\n");
            return;
        }  

        Matrix* grad_B = matrix_scalar_product(loss, learning_rate);
        if (grad_B == NULL) {
            printf("Error in grad_B\n");
            return;
        }  

        free_matrix(prev_layer_T);
        free_matrix(grad_W_);

        if (network->layers[layer].weights == NULL || network->layers[layer].bias == NULL) {
            printf("Error: Failed to update weights or bias.\n");
            return;
        }

        network->layers[layer].weights = matrix_sub(network->layers[layer].weights, grad_W);
        network->layers[layer].bias = matrix_sub(network->layers[layer].bias, grad_B);

        free_matrix(grad_W);
        free_matrix(grad_B);

        if (layer <= 1) 
            continue;
            
        Matrix* transposed_weights = matrix_T(network->layers[layer].weights);

        if (transposed_weights == NULL) {
            printf("Error in transposed_weights\n");
            return;
        }

        Matrix* prev_error = matrix_product(transposed_weights, loss);

        if (prev_error == NULL) {
            printf("Error in prev_error\n");
            return;
        }

        Matrix* prev_layer_RELU_der = RELU_der_matrix(network->layers[layer - 1].nodes);

        if (prev_layer_RELU_der == NULL) {
            printf("Error in prev_layer_RELU_der\n");
            return;
        }

        Matrix* temp_prev_error = matrix_linear_product(prev_error, prev_layer_RELU_der);

        if (temp_prev_error == NULL) {
            printf("Error in temp_prev_error\n");
            return;
        }

        free_matrix(loss);
        loss = temp_prev_error;

        free_matrix(prev_error);
        free_matrix(prev_layer_RELU_der);
        free_matrix(transposed_weights);
    }

    free_matrix(loss);
}

void forward_prop(NeuralNetwork* network, Matrix* input) {
    network->layers[0].nodes = input;

    int num_layers = network->num_layers;

    for (int i = 0; i < network->num_layers - 1; i++) {
        Matrix* product_matrix = matrix_product(network->layers[i].weights, network->layers[i].nodes);
        Matrix* new_layer = matrix_sum(product_matrix, network->layers[i + 1].bias);

        free_matrix(product_matrix);

        if (i < num_layers - 2) {
            RELU_matrix(new_layer);
        } else {
            softmax(new_layer);
        }

        free_matrix(network->layers[i + 1].nodes);
        network->layers[i + 1].nodes = new_layer;
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

void softmax(Matrix* matrix) {
    double sum = 0;
    double max = matrix_get(matrix, 0, 0);
    
    for (int i = 1; i < matrix->row; i++) {
        if (matrix_get(matrix, i, 0) > max) {
            max = matrix_get(matrix, i, 0);
        }
    }

    for (int i = 0; i < matrix->row; i++) {
        sum += exp(matrix_get(matrix, i, 0) - max);
    }

    for (int i = 0; i < matrix->row; i++) {
        matrix_set(matrix, i, 0, exp(matrix_get(matrix, i, 0) - max) / sum);
    }
} 

void RELU(double* pointer) {
    if (*pointer < 0) *pointer = 0;
}

void RELU_matrix(Matrix* pointer) {
    int row_count = pointer->row;
    int col_count = pointer->col;

    for (int r = 0; r < row_count; r++) {
        for (int c = 0; c < col_count; c++) {
            RELU(&(pointer->matrix[r][c]));
        }
    }
}

char RELU_der(double pointer) {
    return pointer > 0 ? 1 : 0;
}

Matrix* RELU_der_matrix(Matrix* matrix) {
    int row_count = matrix->row;
    int col_count = matrix->col;

    Matrix* result = matrix_create(row_count, col_count);
    if (result == NULL) return NULL;

    for (int r = 0; r < row_count; r++) {
        for (int c = 0; c < col_count; c++) {
            result->matrix[r][c] = RELU_der(matrix->matrix[r][c]);
        }
    }

    return result;
}

int get_max_output_node_index(NeuralNetwork* network) {
    Matrix* output_layer = network->layers[network->num_layers - 1].nodes;

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
        free_matrix(dataset[i]);
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
        Layers layer = network->layers[i];

        int rows = layer.nodes->row;
        int cols = layer.nodes->col;
        fwrite(&rows, sizeof(int), 1, file);
        fwrite(&cols, sizeof(int), 1, file);

        if (layer.weights != NULL) {
            fwrite(&(layer.weights->row), sizeof(int), 1, file);
            fwrite(&(layer.weights->col), sizeof(int), 1, file);
            for (int r = 0; r < layer.weights->row; r++) {
                fwrite(layer.weights->matrix[r], sizeof(double), layer.weights->col, file);
            }
        }

        if (layer.bias != NULL) {
            fwrite(&(layer.bias->row), sizeof(int), 1, file);
            fwrite(&(layer.bias->col), sizeof(int), 1, file);
            fwrite(layer.bias->matrix[0], sizeof(double), layer.bias->row * layer.bias->col, file);
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

    network->layers = malloc(sizeof(Layers) * network->num_layers);

    for (int i = 0; i < network->num_layers; i++) {
        Layers* layer = &(network->layers[i]);

        int rows, cols;
        fread(&rows, sizeof(int), 1, file);
        fread(&cols, sizeof(int), 1, file);
        layer->nodes = matrix_create(rows, cols);

        if (i < network->num_layers - 1) {
            fread(&rows, sizeof(int), 1, file);
            fread(&cols, sizeof(int), 1, file);
            layer->weights = matrix_create(rows, cols);
            for (int r = 0; r < rows; r++) {
                fread(layer->weights->matrix[r], sizeof(double), cols, file);
            }
        }

        if (i > 0) {
            fread(&rows, sizeof(int), 1, file);
            fread(&cols, sizeof(int), 1, file);
            layer->bias = matrix_create(rows, cols);
            fread(layer->bias->matrix[0], sizeof(double), rows * cols, file);
        }
    }

    fclose(file);
    return network;
}

