#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include "segmentation.h"

#define STB_IMAGE_IMPLEMENTATION
#include "/Users/cristiano/stb_image/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "/Users/cristiano/stb_image/stb_image_write.h"


int main() {
    original_img = stbi_load("/Users/cristiano/Desktop/Expression.png", &width, &height, &channels, 0);
    pixel_threshold = 50;
    size = width * height * channels;
    visited_pixels = (bool*) calloc(width * height, sizeof(bool));

    if(original_img == NULL) {
        printf("Error in loading image.\n");
        exit(1);
    }

    printf("Noise reduction...\n");

    unsigned char* image_noice_reduced = reduce_noise(original_img);

    printf("Finding the edges...\n");

    unsigned char* image_edges_foud = find_edges(image_noice_reduced);

    free(image_noice_reduced);

    recognize_object(image_edges_foud);

    free(image_edges_foud);

    stbi_write_png("/Users/cristiano/Desktop/precessed_image_final.png", width, height, channels, original_img, (width * channels));

    stbi_image_free(original_img);

    return 0;
}

unsigned char* find_edges(unsigned char* input_image) {
    unsigned char* processed_image = (unsigned char*) calloc(size, sizeof(unsigned char));

    if(processed_image == NULL) {
        printf("Memory Error.\n");
        exit(1);
    }

    for (unsigned long i = 0; i < size; i += channels) {
        apply_filter_pixel(input_image, input_image + i, processed_image + i);
    }

    stbi_write_png("/Users/cristiano/Desktop/Debug/precessed_image_edges.png", width, height, channels, processed_image, (width * channels));

    return processed_image;
}

unsigned char* reduce_noise(unsigned char* input_image) {
    unsigned char* processed_image = (unsigned char*) malloc(sizeof(unsigned char) * size);

    if(processed_image == NULL) {
        printf("Memory Error.\n");
        exit(1);
    }

    for (unsigned long i = 0; i < size; i += channels) {
        reduce_noise_pixel(input_image + i, processed_image + i);
    }

    stbi_write_png("/Users/cristiano/Desktop/Debug/precessed_image_noise.png", width, height, channels, processed_image, (width * channels));

    return processed_image;
}

void recognize_object(unsigned char* input_image) {
    for (int i = 0; i < size; i += channels) {
        visit_pixel(input_image, input_image + i);
    }

    stbi_write_png("/Users/cristiano/Desktop/Debug/precessed_image_visited.png", width, height, channels, input_image, (width * channels));

    printf("Recognized objects: %d\n", box_array_index);

    for (int i = 0; i < box_array_index; i++) {
        draw_rectangle(&box_array[i], original_img);
    }

}

void apply_filter_pixel(unsigned char* image, unsigned char* pixel_pointer, unsigned char* pixel_pointer_output) {
    long r_h = 0;
    long g_h = 0;
    long b_h = 0;
  
    long r_v = 0;
    long g_v = 0;
    long b_v = 0;

    int pixel_pos = (pixel_pointer - image) / channels;
    int pixel_pos_x = pixel_pos % width;
    int pixel_pos_y = pixel_pos / width;

    if (pixel_pos_x < 1 || pixel_pos_x >= width - 1 || pixel_pos_y < 1 || pixel_pos_y >= height - 1)
        return;

    for (char row = -1; row <= 1; row++) {
        for (char col = -1; col <= 1; col++) {
            int new_pixel_x = pixel_pos_x + col;
            int new_pixel_y = pixel_pos_y + row;

            unsigned char* new_coord = image + ((new_pixel_y * width + new_pixel_x) * channels);

            unsigned char pixel_r = *new_coord;
            unsigned char pixel_g = *(new_coord + 1);
            unsigned char pixel_b = *(new_coord + 2);

            r_h += kernel_h[row + 1][col + 1] * pixel_r;
            g_h += kernel_h[row + 1][col + 1] * pixel_g;
            b_h += kernel_h[row + 1][col + 1] * pixel_b;
            
            r_v += kernel_v[row + 1][col + 1] * pixel_r;
            g_v += kernel_v[row + 1][col + 1] * pixel_g;
            b_v += kernel_v[row + 1][col + 1] * pixel_b;
        }
    }

    int new_r_channel = sqrt((r_h * r_h) + (r_v * r_v));
    int new_g_channel = sqrt((g_h * g_h) + (g_v * g_v));
    int new_b_channel = sqrt((b_h * b_h) + (b_v * b_v));

    limit_value(&new_r_channel);
    limit_value(&new_g_channel);
    limit_value(&new_b_channel);

    pixel_apply_color(pixel_pointer_output, (unsigned char)new_r_channel, (unsigned char)new_g_channel, (unsigned char)new_b_channel);

    if (channels > 3) 
        *(pixel_pointer_output + 3) = *pixel_pointer;

    pixel_RELU(pixel_pointer_output);
}

void reduce_noise_pixel(unsigned char* pixel_pointer, unsigned char* pixel_pointer_output) {
    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;

    int pixel_pos = (pixel_pointer - original_img) / channels;
    int pixel_pos_x = pixel_pos % width;
    int pixel_pos_y = pixel_pos / width;

    if (pixel_pos_x < 1 || pixel_pos_x >= width - 1 || pixel_pos_y < 1 || pixel_pos_y >= height - 1) {
        pixel_apply_color(pixel_pointer_output, *pixel_pointer, *(pixel_pointer + 1), *(pixel_pointer + 2));

        if (channels > 3) 
            *(pixel_pointer_output + 3) = *(pixel_pointer + 3);

        return;
    }

    for (char row = -1; row <= 1; row++) {
        for (char col = -1; col <= 1; col++) {
            int new_pixel_x = pixel_pos_x + col;
            int new_pixel_y = pixel_pos_y + row;

            unsigned char* new_coord = original_img + ((new_pixel_y * width + new_pixel_x) * channels);

            unsigned char pixel_r = *new_coord;
            unsigned char pixel_g = *(new_coord + 1);
            unsigned char pixel_b = *(new_coord + 2);

            r += gaussian_kernel[row + 1][col + 1] * pixel_r;
            g += gaussian_kernel[row + 1][col + 1] * pixel_g;
            b += gaussian_kernel[row + 1][col + 1] * pixel_b;
        }
    }

    limit_value_float(&r);
    limit_value_float(&g);
    limit_value_float(&b);

    pixel_apply_color(pixel_pointer_output, (unsigned char)r, (unsigned char)g, (unsigned char)b);

    if (channels > 3) 
        *(pixel_pointer_output + 3) = *(pixel_pointer + 3);
}

void visit_pixel(unsigned char* input_image, unsigned char* pixel_pointer) {
    int start_index = pixel_pointer - input_image;

    if (!pixel_isActive(pixel_pointer) || visited_pixels[start_index / channels])
        return;

    initialize_box();
    int stack_capacity = width * height;
    int* stack = (int*) calloc(stack_capacity, sizeof(int));

    if (stack == NULL) {
        printf("Error: Failed to allocate memory for stack.\n");
        return;
    }

    int stack_top = 0;

    stack[stack_top++] = start_index;
    visited_pixels[start_index / channels] = true;

    int color[3] = {
        rand() % 255,
        rand() % 255,
        rand() % 255,
    };

    while (stack_top > 0) {
        int i = stack[--stack_top];
        unsigned char* current_pixel = input_image + i;

        if (!pixel_isActive(current_pixel)) {
            continue;
        }

        box_array[box_array_index].pixels_length++;
        refresh_box_value(pixel_get_coord(input_image, current_pixel));
        pixel_apply_color(current_pixel, color[0], color[1], color[2]);

        int pixel_pos = i / channels;
        int pixel_pos_x = pixel_pos % width;
        int pixel_pos_y = pixel_pos / width;

        for (char row = -1; row <= 1; row++) {
            for (char col = -1; col <= 1; col++) {
                int new_pixel_x = pixel_pos_x + col;
                int new_pixel_y = pixel_pos_y + row;

                if (new_pixel_x < 0 || new_pixel_x >= width || new_pixel_y < 0 || new_pixel_y >= height) {
                    continue;
                }

                int new_pixel_offset = (new_pixel_y * width + new_pixel_x);

                if(!visited_pixels[new_pixel_offset]) {
                    stack[stack_top++] = new_pixel_offset * channels;
                    visited_pixels[new_pixel_offset] = true;
                }
            }
        }
    }


    free(stack);

    if(box_array[box_array_index].pixels_length > 50 && box_array_index < box_array_size - 1)
        box_array_index++;
}

void pixel_RELU(unsigned char* pixel_pointer) {
    bool active = pixel_isActive(pixel_pointer);
    unsigned char v = active ? 255 : 0;

    pixel_apply_color(pixel_pointer, v, v, v);
}

bool pixel_isActive(unsigned char* coord) {
    unsigned char red = *(coord);
    unsigned char green = *(coord + 1);
    unsigned char blue = *(coord + 2);

    unsigned char lum = (red + green + blue) / 3;

    return lum > pixel_threshold;
}

void limit_value(int* value) {
    if (*value < 0)
        *value = 0;
    else if (*value > 255)
        *value = 255;
}

void limit_value_float(float* value) {
    if (*value < 0)
        *value = 0;
    else if (*value > 255)
        *value = 255;
}

int* pixel_get_coord(unsigned char* image, unsigned char* pointer) {
    long offset = pointer - image;
    int* a = malloc(2 * sizeof(int));
    *a = (offset / channels) % width,
    *(a + 1) = (offset / channels) / width;

    return a;
}

void pixel_apply_color(unsigned char* pointer, unsigned char r, unsigned char g, unsigned char b){
    *(pointer) = r;
    *(pointer + 1) = g;
    *(pointer + 2) = b;
}

void initialize_box() {
    if (box_array_index > box_array_size - 1)
        return;
    
    Box* box = &box_array[box_array_index];
    box->minY = INT32_MAX;
    box->minX = INT32_MAX;
    box->maxY = INT32_MIN;
    box->maxX = INT32_MIN;
    box->pixels_length = 0;
}

void refresh_box_value(int* coord) {
    int x = *coord;
    int y = *(coord + 1);

    Box* box = &box_array[box_array_index];
    box->minX = box->minX < x ? box->minX : x;
    box->maxX = box->maxX > x ? box->maxX : x;
    box->minY = box->minY < y ? box->minY : y;
    box->maxY = box->maxY > y ? box->maxY : y;
}

void draw_rectangle(Box* current_box, unsigned char* image) {
    unsigned char* top_left = image + ((current_box->minY * width + current_box->minX) * channels);
    unsigned char* top_right = image + ((current_box->minY * width + current_box->maxX) * channels);
    unsigned char* bottom_left = image + ((current_box->maxY * width + current_box->minX) * channels);
    unsigned char* bottom_right = image + ((current_box->maxY * width + current_box->maxX) * channels);

    draw_line(top_left, top_right, image);
    draw_line(bottom_left, bottom_right, image);
    draw_line(top_left, bottom_left, image);
    draw_line(top_right, bottom_right, image);
}

void draw_line(unsigned char* pixel1, unsigned char* pixel2, unsigned char* image) {
    int x1 = (pixel1 - image) / channels % width;
    int y1 = (pixel1 - image) / channels / width;
    int x2 = (pixel2 - image) / channels % width;
    int y2 = (pixel2 - image) / channels / width;

    int dx = abs(x2 - x1);
    int dy = abs(y2 - y1);
    
    int sx = (x1 < x2) ? 1 : -1;
    int sy = (y1 < y2) ? 1 : -1;

    int err = dx - dy;

    while (x1 != x2 || y1 != y2) {
        unsigned char* p = image + channels * (y1 * width + x1);

        *p = 255;
        *(p + 1) = 0; 
        *(p + 2) = 0;

        int e2 = 2 * err;
        if (e2 > -dy) {
            err -= dy;
            x1 += sx;
        }
        if (e2 < dx) {
            err += dx;
            y1 += sy;
        }
    }

    unsigned char* p = image + channels * (y2 * width + x2);
    *p = 255; 
    *(p + 1) = 0;  
    *(p + 2) = 0; 
}

