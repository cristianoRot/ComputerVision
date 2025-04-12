
#include <stdlib.h>
#include <stdio.h>
#include "matrix.h"

Matrix* matrix_create(int row, int col) {
    Matrix* matrix = (Matrix*) malloc(sizeof(Matrix));

    if (matrix == NULL) {
        printf("Error during matrix creation.\n");
        return NULL;
    }

    matrix->matrix = (double**) malloc(row * sizeof(double*));

    for (int i = 0; i < row; i++) {
        matrix->matrix[i] = (double*) malloc(col * sizeof(double));
    }
    
    matrix->row = row;
    matrix->col = col;

    return matrix;
}

double matrix_get(Matrix* matrix, int r, int c) {
    return matrix->matrix[r][c];
}

void matrix_set(Matrix* matrix, int r, int c, double v) {
    matrix->matrix[r][c] = v;
}

Matrix* matrix_T(Matrix* matrix) {
    if(matrix == NULL) {
        printf("Unable to transpose.");  
        return NULL;
    }

    Matrix* matrix_T = matrix_create(matrix->col, matrix->row);

    if (matrix_T == NULL) return NULL;

    for(int r = 0; r < matrix->row; r++){
        for(int c = 0; c < matrix->col; c++){
            matrix_T->matrix[c][r] = matrix->matrix[r][c];
        }
    }

    return matrix_T;
}

Matrix* matrix_sub(Matrix* m1, Matrix* m2) { 
    if(m1 == NULL || m2 == NULL || m1->row != m2->row || m1->col != m2->col) {
        printf("Unable to subtract");  
        return NULL;
    } 

    Matrix* matrix = matrix_create(m1->row, m1->col);
    if (matrix == NULL) return NULL;
    
    for(int r = 0; r < m1->row; r++){
        for(int c = 0; c < m1->col; c++){
            matrix->matrix[r][c] = m1->matrix[r][c] - m2->matrix[r][c];
        }
    }

    return matrix;
}

Matrix* matrix_sum(Matrix* m1, Matrix* m2) {
    // Controllo delle dimensioni
    if (m1 == NULL || m2 == NULL || m1->row != m2->row || m1->col != m2->col) {
        printf("Unable to sum: matrices have incompatible dimensions.\n");
        return NULL;
    }

    // Creazione della matrice risultato
    Matrix* matrix = matrix_create(m1->row, m1->col);
    if (matrix == NULL) return NULL;
    
    // Esegui la somma elemento per elemento
    for (int r = 0; r < m1->row; r++) {
        for (int c = 0; c < m1->col; c++) {
            matrix->matrix[r][c] = m1->matrix[r][c] + m2->matrix[r][c];
        }
    }

    return matrix;
}

Matrix* matrix_product(Matrix* m1, Matrix* m2) {
    if(m1 == NULL || m2 == NULL || m1->col != m2->row) {
        printf("Unable to product");  
        return NULL;
    }

    Matrix* res = matrix_create(m1->row, m2->col);
    if (res == NULL) return NULL;
    
    for(int r = 0; r < m1->row; r++){
        for(int c = 0; c < m2->col; c++){
            double x = 0;
            for (int n = 0; n < m1->col; n++) {
                x += m1->matrix[r][n] * m2->matrix[n][c];
            }
            res->matrix[r][c] = x;
        }
    }

    return res;
}

Matrix* matrix_scalar_product(Matrix* m1, double scalar) {
    if(m1 == NULL) {
        printf("Unable to scalar product");  
        return NULL;
    }

    Matrix* result = matrix_create(m1->row, m1->col);
    if (result == NULL) return NULL;
    
    for (int r = 0; r < m1->row; r++) {
        for (int c = 0; c < m1->col; c++) {
            result->matrix[r][c] = m1->matrix[r][c] * scalar;
        }
    }
    return result;
}

Matrix* matrix_linear_product(Matrix* m1, Matrix* m2) {
    if(m1 == NULL || m2 == NULL || m1->row != m2->row || m1->col != m2->col) {
        printf("Unable to linear product");  
        return NULL;
    }

    Matrix* result = matrix_create(m1->row, m1->col);
    if (result == NULL) return NULL;

    for (int r = 0; r < m1->row; r++) {
        for (int c = 0; c < m1->col; c++) {
            result->matrix[r][c] = m1->matrix[r][c] * m2->matrix[r][c];
        }
    }
    return result;
}

void matrix_print(Matrix* matrix) {
    for(int r = 0; r < matrix->row; r++){
        for(int c = 0; c < matrix->col; c++){
            printf("%f ", matrix->matrix[r][c]);
        }
        
        printf("\n");
    }
}

void free_matrix(Matrix* mat) {
    if (mat != NULL && mat->matrix != NULL) {
        // Libera ogni riga della matrice
        for (int i = 0; i < mat->row; i++) {
            if (mat->matrix[i] != NULL) {
                free(mat->matrix[i]);
                mat->matrix[i] = NULL;
            }
        }
        free(mat->matrix);
        mat->matrix = NULL;
    }
    free(mat);
}