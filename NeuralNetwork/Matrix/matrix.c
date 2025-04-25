#include "matrix.h"
#include <Accelerate/Accelerate.h>
#include <stdlib.h>

#ifndef ALIGNMENT
#define ALIGNMENT 64
#endif

Matrix* matrix_create(int row, int col) {
    if (row <= 0 || col <= 0) {
        fprintf(stderr, "Error in matrix_create: Matrix dimensions must be positive (%d, %d).\n", row, col);
        return NULL;
    }

    Matrix* matrix = (Matrix*) malloc(sizeof(Matrix));
    if (matrix == NULL) {
        perror("Error in matrix_create: Failed to allocate matrix struct");
        return NULL;
    }

    size_t total_bytes = (size_t)row * col * sizeof(double);
    // matrix->data = aligned_alloc(ALIGNMENT, total_bytes);
    void* ptr = NULL;
    if (posix_memalign(&ptr, ALIGNMENT, total_bytes) != 0) {
        perror("Error in matrix_create: Failed to allocate aligned data");
        free(matrix);
        return NULL;
    }
    matrix->data = ptr;

    if (!matrix->data) {
        perror("Error in matrix_create: Failed to allocate matrix data");
        free(matrix);
        return NULL;
    }

    matrix->row = row;
    matrix->col = col;

    return matrix;
}

double matrix_get(const Matrix* matrix, int r, int c) {
     if (matrix == NULL || r < 0 || r >= matrix->row || c < 0 || c >= matrix->col) {
        fprintf(stderr, "Error in matrix_get: Out of bounds access (%d, %d) for matrix (%dx%d) or NULL matrix.\n",
                r, c, matrix ? matrix->row : 0, matrix ? matrix->col : 0);
        exit(EXIT_FAILURE);
    }
    return matrix->data[r * matrix->col + c];
}

void matrix_set(Matrix* matrix, int r, int c, double v) {
    if (matrix == NULL || r < 0 || r >= matrix->row || c < 0 || c >= matrix->col) {
        fprintf(stderr, "Error in matrix_set: Out of bounds access (%d, %d) for matrix (%dx%d) or NULL matrix.\n",
                r, c, matrix ? matrix->row : 0, matrix ? matrix->col : 0);
        exit(EXIT_FAILURE);
        return;
    }
    matrix->data[r * matrix->col + c] = v;
}

Matrix* matrix_random(int rows, int cols) {
    Matrix* m = matrix_create(rows, cols);
    if (!m) return NULL;

    static int seeded = 0;
    if (!seeded) {
        srand((unsigned)time(NULL));
        seeded = 1;
    }

    int total_elements = rows * cols;
    for (int i = 0; i < total_elements; ++i) {
        m->data[i] = (double)rand() / ((double)RAND_MAX + 1.0);
    }

    return m;
}

Matrix* matrix_T(const Matrix* matrix) {
    if(matrix == NULL) {
        fprintf(stderr, "Error in matrix_T: Unable to transpose NULL matrix.\n");
        return NULL;
    }

    Matrix* matrix_T_result = matrix_create(matrix->col, matrix->row);
    if (matrix_T_result == NULL) return NULL;

    for (int r = 0; r < matrix->row; ++r) {
        for (int c = 0; c < matrix->col; ++c) {
            matrix_T_result->data[c * matrix->row + r] = matrix->data[r * matrix->col + c];
        }
    }
    return matrix_T_result;
}

Matrix* matrix_sub(const Matrix* m1, const Matrix* m2) {
    if(m1 == NULL || m2 == NULL || m1->row != m2->row || m1->col != m2->col) {
        fprintf(stderr, "Error in matrix_sub: Matrices have incompatible dimensions or are NULL.\n");
        return NULL;
    }

    int total_elements = m1->row * m1->col;
    Matrix* result = matrix_create(m1->row, m1->col);
    if (result == NULL) return NULL;

    memcpy(result->data, m1->data, total_elements * sizeof(double));
    cblas_daxpy(total_elements, -1.0, m2->data, 1, result->data, 1);

    return result;
}

Matrix* matrix_sum(const Matrix* m1, const Matrix* m2) {
    if (m1 == NULL || m2 == NULL || m1->row != m2->row || m1->col != m2->col) {
        fprintf(stderr, "Error in matrix_sum: Matrices have incompatible dimensions or are NULL.\n");
        return NULL;
    }

    int total_elements = m1->row * m1->col;
    Matrix* result = matrix_create(m1->row, m1->col);
    if (result == NULL) return NULL;

    memcpy(result->data, m1->data, total_elements * sizeof(double));
    cblas_daxpy(total_elements, 1.0, m2->data, 1, result->data, 1);

    return result;
}

Matrix* matrix_product(const Matrix* m1, const Matrix* m2) {
    if(m1 == NULL || m2 == NULL || m1->col != m2->row) {
        fprintf(stderr, "Error in matrix_product: Matrices have incompatible dimensions for multiplication or are NULL. (%dx%d) * (%dx%d)\n",
                m1 ? m1->row : 0, m1 ? m1->col : 0, m2 ? m2->row : 0, m2 ? m2->col : 0);
        return NULL;
    }

    Matrix* result = matrix_create(m1->row, m2->col);
    if (result == NULL) return NULL;

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m1->row, m2->col, m1->col,
                1.0, m1->data, m1->col,
                m2->data, m2->col,
                0.0, result->data, result->col);

    return result;
}

Matrix* matrix_scalar_product(const Matrix* m1, double scalar) {
    if(m1 == NULL) {
        fprintf(stderr, "Error in matrix_scalar_product: Unable to perform scalar product on NULL matrix.\n");
        return NULL;
    }

    int total_elements = m1->row * m1->col;
    Matrix* result = matrix_create(m1->row, m1->col);
    if (result == NULL) return NULL;

    memcpy(result->data, m1->data, total_elements * sizeof(double));
    cblas_dscal(total_elements, scalar, result->data, 1);

    return result;
}

Matrix* matrix_linear_product(const Matrix* m1, const Matrix* m2) {
    if(m1 == NULL || m2 == NULL || m1->row != m2->row || m1->col != m2->col) {
        fprintf(stderr, "Error in matrix_linear_product: Matrices have incompatible dimensions for element-wise product or are NULL.\n");
        return NULL;
    }

    int total_elements = m1->row * m1->col;
    Matrix* result = matrix_create(m1->row, m1->col);
    if (result == NULL) return NULL;

    for (int i = 0; i < total_elements; i++) {
        result->data[i] = m1->data[i] * m2->data[i];
    }
    return result;
}

Matrix* matrix_column_sum(const Matrix* matrix) {
    if (matrix == NULL) {
        fprintf(stderr, "Error in matrix_column_sum: Unable to calculate column sum on NULL matrix.\n");
        return NULL;
    }

    int rows = matrix->row;
    int cols = matrix->col;

    Matrix* result = matrix_create(rows, 1);
    if (!result) return NULL;

    for (int i = 0; i < rows; i++) {
        double sum = 0.0;
        for (int j = 0; j < cols; j++) {
            sum += matrix->data[i * cols + j];
        }
        result->data[i * 1 + 0] = sum;
    }

    return result;
}

void matrix_copy(const Matrix* m1, Matrix* m2) {
    if(m1 == NULL || m2 == NULL || m1->row != m2->row || m1->col != m2->col) {
        fprintf(stderr, "Error in matrix_copy: Matrices have incompatible dimensions or are NULL.\n");
        return;
    }

    int total_elements = m1->row * m1->col;
    memcpy(m2->data, m1->data, total_elements * sizeof(double));
}

void matrix_print(const Matrix* data) {
    if (data == NULL) {
        printf("NULL Matrix\n");
        return;
    }

    for(int r = 0; r < data->row; r++){
        for(int c = 0; c < data->col; c++){
            printf("%f ", data->data[r * data->col + c]);
        }
        printf("\n");
    }
}

void matrix_free(Matrix* mat) {
    if (mat != NULL) {
        if (mat->data != NULL) {
            free(mat->data);
            mat->data = NULL;
        }
        free(mat);
    }
}