#include "dataset_parser.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>
#define STB_IMAGE_IMPLEMENTATION
#include "/Users/cristiano/stb_image/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "/Users/cristiano/stb_image/stb_image_write.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "/Users/cristiano/stb_image/stb_image_resize2.h"

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

            Matrix* img = matrix_create(canvas_size * canvas_size, 1);
            for (int i = 0; i < canvas_size * canvas_size; ++i) {
                img->data[i][0] = resized[i] / 255.0;
            }
            free(resized);

            Matrix* label = matrix_create(num_classes, 1);
            for (int i = 0; i < num_classes; ++i) label->data[i][0] = 0.0;
            label->data[c][0] = 1.0;

            ds->X[idx] = img;
            ds->Y[idx] = label;
            idx++;
        }
        closedir(d);
    }
    ds->N = idx;
    return idx;
}

int load_csv(Dataset* ds, const char* csv_path, int num_features, int num_classes) {
    if (!ds || !csv_path) return -1;
    FILE* f = fopen(csv_path, "r");
    if (!f) return -1;

    char line[4096];
    int count = 0;
    while (fgets(line, sizeof(line), f)) {
        char* ptr = line;
        while (*ptr == ' ' || *ptr == '\t') ptr++;
        if (*ptr && *ptr != '\n') count++;
    }
    if (count == 0) { fclose(f); return -1; }
    ds->X = malloc(sizeof(Matrix*) * count);
    ds->Y = malloc(sizeof(Matrix*) * count);
    rewind(f);

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
            feat->data[i][0] = strtod(token, NULL);
            token = strtok(NULL, ",");
            i++;
        }
        int cls = token ? atoi(token) : 0;
        for (int j = 0; j < num_classes; ++j) lab->data[j][0] = (j == cls);

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