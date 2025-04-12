/*
    Segmentation Header
*/


typedef struct {
    int pixels_length;
    int minY, maxY;
    int minX, maxX; 
} Box;

char kernel_v[3][3] = {
  {-1, -2, -1},
  {0, 0, 0},
  {1, 2, 1}
};

char kernel_h[3][3] = {
  {-1, 0, 1},
  {-2, 0, 2},
  {-1, 0, 1}
};

float gaussian_kernel[3][3] = {
    {1/16.0, 2/16.0, 1/16.0},
    {2/16.0, 4/16.0, 2/16.0},
    {1/16.0, 2/16.0, 1/16.0}
};

int width, height, channels, size;
int pixel_threshold;
unsigned char* original_img;

bool* visited_pixels;

const int box_array_size = 1000;
Box box_array[box_array_size];
int box_array_index = 0;

unsigned char* find_edges(unsigned char* input_image);

unsigned char* reduce_noise(unsigned char* input_image);

void recognize_object(unsigned char* input_image);

void apply_filter_pixel(unsigned char* image, unsigned char* pixel_pointer, unsigned char* pixel_pointer_output);

void reduce_noise_pixel(unsigned char* pixel_pointer, unsigned char* pixel_pointer_output);

void visit_pixel(unsigned char* input_image, unsigned char* pixel_pointer);

void limit_value(int* value);

void limit_value_float(float* value);

bool pixel_isActive(unsigned char* coord);

int* pixel_get_coord(unsigned char* image, unsigned char* pointer);

void refresh_box_value(int* coord);

void pixel_RELU(unsigned char* pixel_pointer);

void pixel_apply_color(unsigned char* pointer, unsigned char r, unsigned char g, unsigned char b);

void draw_rectangle(Box* current_box, unsigned char* image);

void draw_line(unsigned char* pixel1, unsigned char* pixel2, unsigned char* image);

void initialize_box();