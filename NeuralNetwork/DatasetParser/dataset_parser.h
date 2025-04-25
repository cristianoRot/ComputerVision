#ifndef DATASET_PARSER_H
#define DATASET_PARSER_H

#include "../Matrix/matrix.h"

typedef struct {
    Matrix** X;
    Matrix** Y;
    int N;
} Dataset;

/**
 * @brief Load an image dataset where each subdirectory under root_path is a class label.
 * @param ds Pointer to Dataset to populate.
 * @param root_path Path to root folder containing one subdirectory per class.
 * @param canvas_size Image dimension to resize (width and height).
 * @param num_classes Number of class subdirectories.
 * @return Total number of examples loaded, or -1 on error.
 */
int load_image_folder(Dataset* ds, const char* root_path, int canvas_size, int num_classes);

int load_csv_label_first(Dataset* ds, const char* csv_path, int num_features, int num_classes);

/**
 * @brief Load a CSV dataset with features in the first num_features columns and label in the last column.
 * @param ds Pointer to Dataset to populate.
 * @param csv_path Path to the CSV file.
 * @param num_features Number of feature columns.
 * @param num_classes Number of distinct classes for one-hot encoding.
 * @return Total number of examples loaded, or -1 on error.
 */
int load_csv_label_last(Dataset* ds, const char* csv_path, int num_features, int num_classes);

/**
 * @brief Free all memory associated with the Dataset.
 * @param ds Pointer to Dataset to free.
 */
void free_dataset(Dataset* ds);

void dataset_print(const Dataset* ds);

int save_matrix_as_image(const Matrix* matrix, const char* filepath);

#endif