// matrix.h

#ifndef MATRIX_H
#define MATRIX_H

typedef struct {
    double** matrix;
    int row;
    int col;
} Matrix;

Matrix* matrix_create(int row, int col);

double matrix_get(Matrix* matrix, int r, int c);

void matrix_set(Matrix* matrix, int r, int c, double v);

Matrix* matrix_T(Matrix* matrix);

Matrix* matrix_sub(Matrix* m1, Matrix* m2);

Matrix* matrix_sum(Matrix* m1, Matrix* m2);

Matrix* matrix_product(Matrix* m1, Matrix* m2);

Matrix* matrix_scalar_product(Matrix* m1, double scalar);

Matrix* matrix_linear_product(Matrix* m1, Matrix* m2);

void matrix_print(Matrix* matrix);

void free_matrix(Matrix* mat);

#endif
