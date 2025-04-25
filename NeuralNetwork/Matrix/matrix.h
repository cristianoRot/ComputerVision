// matrix.h

#ifndef MATRIX_H
#define MATRIX_H

typedef struct {
    int row;
    int col;
    double *data;
} Matrix;

Matrix* matrix_create(int row, int col);

double matrix_get(const Matrix* data, int r, int c);

void matrix_set(Matrix* data, int r, int c, double v);

Matrix* matrix_random(int rows, int cols);

Matrix* matrix_T(const Matrix* data);

Matrix* matrix_sub(const Matrix* m1, const Matrix* m2);

Matrix* matrix_sum(const Matrix* m1, const Matrix* m2);

Matrix* matrix_product(const Matrix* m1, const Matrix* m2);

Matrix* matrix_scalar_product(const Matrix* m1, double scalar);

Matrix* matrix_linear_product(const Matrix* m1, const Matrix* m2);

Matrix* matrix_column_sum(const Matrix* mat);

void matrix_copy(const Matrix* m1, Matrix* m2);

void matrix_print(const Matrix* data);

void matrix_free(Matrix* mat);

#endif // MATRIX_H
