#include "dataset_parser.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>
#include <errno.h>

#define STB_IMAGE_IMPLEMENTATION
#include "../../stb_image/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../stb_image/stb_image_write.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "../../stb_image/stb_image_resize2.h"

int load_image_folder(Dataset* ds, const char* root_path, int canvas_size, int num_classes) {
    if (!ds || !root_path || canvas_size <= 0 || num_classes <= 0) return -1;

    // count total images per class
    int total = 0;
    char path[1024];
    for (int c = 0; c < num_classes; ++c) {
        snprintf(path, sizeof(path), "%s/%d", root_path, c);
        DIR* d = opendir(path);
        if (!d) return -1;
        struct dirent* ent;
        while ((ent = readdir(d)) != NULL) {
            if (ent->d_name[0] == '.') continue;
            char file[1024];
            snprintf(file, sizeof(file), "%s/%s", path, ent->d_name);
            struct stat st;
            if (stat(file, &st) == 0 && S_ISREG(st.st_mode)) total++;
        }
        closedir(d);
    }
    if (total == 0) return -1;

    ds->X = malloc(sizeof(Matrix*) * total);
    ds->Y = malloc(sizeof(Matrix*) * total);
    if (!ds->X || !ds->Y) return -1;

    // load images and labels
    int idx = 0;
    for (int c = 0; c < num_classes; ++c) {
        snprintf(path, sizeof(path), "%s/%d", root_path, c);
        DIR* d = opendir(path);
        if (!d) break;
        struct dirent* ent;
        while ((ent = readdir(d)) != NULL) {
            if (ent->d_name[0] == '.') continue;
            char file[1024];
            snprintf(file, sizeof(file), "%s/%s", path, ent->d_name);
            struct stat st;
            if (stat(file, &st) != 0 || !S_ISREG(st.st_mode)) continue;

            int w, h, channels;
            unsigned char* data = stbi_load(file, &w, &h, &channels, 0); // load original channels
            if (!data) continue;

            unsigned char* gray_orig = malloc(w * h);

            if (!gray_orig) {
                stbi_image_free(data);
                continue;
            }
            
            for (int i = 0; i < w * h; ++i) {
                int sum = 0;
                for (int ch = 0; ch < channels; ++ch) {
                    sum += data[i * channels + ch];
                }
                gray_orig[i] = (unsigned char)(sum / channels);
            }
            stbi_image_free(data);

            unsigned char* resized = malloc(canvas_size * canvas_size);
            stbir_resize_uint8_linear(gray_orig, w, h, 0,
                                      resized, canvas_size, canvas_size, 0, 1);
            free(gray_orig);

            int final_num_features = canvas_size * canvas_size;

            Matrix* img_matrix = matrix_create(canvas_size * canvas_size, 1);

            for (int i = 0; i < final_num_features; ++i) {
                img_matrix->data[i] = (double)resized[i] / 255.0;
            }

            free(resized);

            Matrix* label = matrix_create(num_classes, 1);

            for (int i = 0; i < num_classes; ++i) {
                label->data[i] = 0.0;
            }

            label->data[c] = 1.0;

            ds->X[idx] = img_matrix;
            ds->Y[idx] = label;
            idx++;
        }
        closedir(d);
    }
    ds->N = idx;
    return idx;
}

int load_csv_label_first(Dataset* ds, const char* csv_path, int num_features, int num_classes) {
    if (!ds || !csv_path) return -1;
    FILE* f = fopen(csv_path, "r");
    if (!f) return -1;

    char line[4096];
    int count = 0;

    if (!fgets(line, sizeof(line), f)) {
        fclose(f);
        return -1;
    }

    while (fgets(line, sizeof(line), f)) {
        char* ptr = line;
        while (*ptr == ' ' || *ptr == '\t') ptr++;
        if (*ptr && *ptr != '\n' && *ptr != '\r') count++;
    }
    if (count == 0) {
        fclose(f);
        return 0;
    }

    ds->X = malloc(sizeof(Matrix*) * count);
    if (!ds->X) {
        fclose(f);
        return -1;
    }
    ds->Y = malloc(sizeof(Matrix*) * count);
     if (!ds->Y) {
        free(ds->X);
        fclose(f);
        return -1;
    }

    rewind(f);

    if (!fgets(line, sizeof(line), f)) {
         free(ds->X); free(ds->Y);
         fclose(f);
         return -1;
    }

    int idx = 0;

    while (fgets(line, sizeof(line), f)) {
        char* ptr = line;
        while (*ptr == ' ' || *ptr == '\t') ptr++;
        if (!*ptr || *ptr == '\n' || *ptr == '\r') continue;

        Matrix* feat = matrix_create(num_features, 1);
        Matrix* lab  = matrix_create(num_classes, 1);

        if (!feat || !lab) {
            fprintf(stderr, "Error allocating matrices for line %d.\n", idx + 2);
            for (int k = 0; k < idx; ++k) {
                matrix_free(ds->X[k]);
                matrix_free(ds->Y[k]);
            }
            free(ds->X); free(ds->Y);
            matrix_free(feat); matrix_free(lab);
            fclose(f);
            return -1;
        }

        char* token = strtok(line, ",");
        if (!token) {
            fprintf(stderr, "Error parsing line %d: Missing label token.\n", idx + 2);
            matrix_free(feat); matrix_free(lab);
            continue;
        }

        int cls = atoi(token);
        if (cls < 0 || cls >= num_classes) {
             fprintf(stderr, "Error parsing line %d: Invalid class index %d (expected 0-%d).\n", idx + 2, cls, num_classes - 1);
             matrix_free(feat); matrix_free(lab);
             continue;
        }

        token = strtok(NULL, ",");
        int i = 0; // Contatore per le feature

        while (token && i < num_features) {
            feat->data[i] = strtod(token, NULL);
            token = strtok(NULL, ",");
            i++;
        }

        if (i != num_features) {
            fprintf(stderr, "Error parsing line %d: Expected %d features, but found %d.\n", idx + 2, num_features, i);
            matrix_free(feat); matrix_free(lab);
            continue;
        }


        for (int j = 0; j < num_classes; ++j) {
            lab->data[j] = (j == cls) ? 1.0 : 0.0;
        }

        ds->X[idx] = feat;
        ds->Y[idx] = lab;
        idx++; // Incrementa l'indice del dataset
    }

    fclose(f);

    ds->N = idx;
    return idx;
}

int load_csv_label_last(Dataset* ds, const char* csv_path, int num_features, int num_classes) {
    if (!ds || !csv_path) return -1;
    FILE* f = fopen(csv_path, "r");
    if (!f) return -1;

    char line[4096];
    int count = 0;

    // Skip header line
    if (!fgets(line, sizeof(line), f)) {
        fclose(f);
        return -1;
    }

    while (fgets(line, sizeof(line), f)) {
        char* ptr = line;
        while (*ptr == ' ' || *ptr == '\t') ptr++;
        if (*ptr && *ptr != '\n') count++;
    }
    if (count == 0) { fclose(f); return -1; }
    ds->X = malloc(sizeof(Matrix*) * count);
    ds->Y = malloc(sizeof(Matrix*) * count);
    rewind(f);
    
    // Skip header before parsing
    fgets(line, sizeof(line), f);

    int idx = 0;
    while (fgets(line, sizeof(line), f)) {
        char* ptr = line;
        while (*ptr == ' ' || *ptr == '\t') ptr++;
        if (!*ptr || *ptr == '\n') continue;

        Matrix* feat = matrix_create(num_features, 1);
        Matrix* lab  = matrix_create(num_classes, 1);
        char* token = strtok(line, ",");
        int i = 0;

        while (token && i < num_features) {
            feat->data[i] = strtod(token, NULL);
            token = strtok(NULL, ",");
            i++;
        }

        int cls = token ? atoi(token) : 0;

        for (int j = 0; j < num_classes; ++j) {
            lab->data[j] = (j == cls) ? 1.0 : 0.0;
        }

        ds->X[idx] = feat;
        ds->Y[idx] = lab;
        idx++;
    }
    fclose(f);
    ds->N = idx;
    return idx;
}

void free_dataset(Dataset* ds) {
    for (int i = 0; i < ds->N; ++i) {
        matrix_free(ds->X[i]);
        matrix_free(ds->Y[i]);
    }
    free(ds->X);
    free(ds->Y);
    ds->X = ds->Y = NULL;
    ds->N = 0;
}

void dataset_print(const Dataset* ds) {
    if (ds == NULL) {
        printf("Error: Dataset pointer is NULL.\n");
        return;
    }

    printf("--- Dataset Content (%d examples) ---\n", ds->N);

    if (ds->X == NULL || ds->Y == NULL) {
        printf("Warning: Dataset X or Y arrays are NULL.\n");
        return;
    }


    for (int i = 0; i < ds->N; ++i) {
        printf("Example %d:\n", i);

        printf("  Input (X[%d]):\n", i);
        if (ds->X[i] != NULL) {
            matrix_print(ds->X[i]);
        } else {
            printf("    Matrix pointer is NULL.\n");
        }

        printf("  True Label (Y[%d]):\n", i);
        if (ds->Y[i] != NULL) {
            matrix_print(ds->Y[i]);
        } else {
            printf("    Matrix pointer is NULL.\n");
        }

        printf("---\n");
    }

    printf("--- End Dataset Content ---\n");
}

int save_matrix_as_image(const Matrix* matrix, const char* filepath) {
    if (matrix == NULL || matrix->data == NULL) {
        fprintf(stderr, "Error in save_matrix_as_image: Input matrix is NULL or data is NULL.\n");
        return -1;
    }
    if (filepath == NULL || *filepath == '\0') {
         fprintf(stderr, "Error in save_matrix_as_image: Output filepath is invalid (NULL or empty).\n");
         return -1;
    }

    int total_pixels = matrix->row;
    int channels = 1;

    int width = (int)round(sqrt(total_pixels));
    int height = width;

    if (width * height != total_pixels || matrix->col != 1) {
        fprintf(stderr, "Error in save_matrix_as_image: Matrix dimensions (%dx%d) are incompatible with a flattened %dx%d grayscale image (%d total pixels). Check matrix->row matches width*height and matrix->col is 1.\n",
                matrix->row, matrix->col, width, height, total_pixels);
        return -1;
    }

    unsigned char* pixel_buffer = malloc((size_t)total_pixels * channels);
    if (!pixel_buffer) {
        fprintf(stderr, "Error in save_matrix_as_image: Failed to allocate pixel buffer (%s).\n", strerror(errno));
        return -1;
    }

    for (int i = 0; i < total_pixels; ++i) {
        double double_value = matrix->data[i];

        int byte_value = (int)(double_value * 255.0 + 0.5);
        if (byte_value < 0) byte_value = 0;
        if (byte_value > 255) byte_value = 255;

        pixel_buffer[i] = (unsigned char)byte_value;
    }

    int success = stbi_write_png(filepath, width, height, channels, pixel_buffer, width * channels);

    free(pixel_buffer);

    if (!success) {
        fprintf(stderr, "Error in save_matrix_as_image: Failed to write PNG file %s.\n", filepath);
        return -1;
    }

    printf("Matrix successfully saved as grayscale PNG: %s (%dx%d)\n", filepath, width, height);
    return 0;
}