
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "matrix.h"

Matrix* matrix_create(int row, int col) {
    Matrix* data = (Matrix*) malloc(sizeof(Matrix));

    if (data == NULL) {
        printf("Error during data creation.\n");
        return NULL;
    }

    data->data = (double**) malloc(row * sizeof(double*));

    for (int i = 0; i < row; i++) {
        data->data[i] = (double*) malloc(col * sizeof(double));
    }
    
    data->row = row;
    data->col = col;

    return data;
}

double matrix_get(Matrix* data, int r, int c) {
    return data->data[r][c];
}

void matrix_set(Matrix* data, int r, int c, double v) {
    data->data[r][c] = v;
}

Matrix* matrix_random(int rows, int cols) {
    Matrix* m = matrix_create(rows, cols);
    if (!m) return NULL;

    static int seeded = 0;
    if (!seeded) {
        srand((unsigned)time(NULL));
        seeded = 1;
    }

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            m->data[i][j] = (double)rand() / ((double)RAND_MAX + 1.0);
        }
    }

    return m;
}

Matrix* matrix_T(Matrix* data) {
    if(data == NULL) {
        printf("Unable to transpose.");  
        return NULL;
    }

    Matrix* matrix_T = matrix_create(data->col, data->row);

    if (matrix_T == NULL) return NULL;

    for(int r = 0; r < data->row; r++){
        for(int c = 0; c < data->col; c++){
            matrix_T->data[c][r] = data->data[r][c];
        }
    }

    return matrix_T;
}

Matrix* matrix_sub(Matrix* m1, Matrix* m2) { 
    if(m1 == NULL || m2 == NULL || m1->row != m2->row || m1->col != m2->col) {
        printf("Unable to subtract");  
        return NULL;
    } 

    Matrix* data = matrix_create(m1->row, m1->col);
    if (data == NULL) return NULL;
    
    for(int r = 0; r < m1->row; r++){
        for(int c = 0; c < m1->col; c++){
            data->data[r][c] = m1->data[r][c] - m2->data[r][c];
        }
    }

    return data;
}

Matrix* matrix_sum(Matrix* m1, Matrix* m2) {
    // Controllo delle dimensioni
    if (m1 == NULL || m2 == NULL || m1->row != m2->row || m1->col != m2->col) {
        printf("Unable to sum: matrices have incompatible dimensions.\n");
        return NULL;
    }

    // Creazione della matrice risultato
    Matrix* data = matrix_create(m1->row, m1->col);
    if (data == NULL) return NULL;
    
    // Esegui la somma elemento per elemento
    for (int r = 0; r < m1->row; r++) {
        for (int c = 0; c < m1->col; c++) {
            data->data[r][c] = m1->data[r][c] + m2->data[r][c];
        }
    }

    return data;
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
                x += m1->data[r][n] * m2->data[n][c];
            }
            res->data[r][c] = x;
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
            result->data[r][c] = m1->data[r][c] * scalar;
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
            result->data[r][c] = m1->data[r][c] * m2->data[r][c];
        }
    }
    return result;
}

Matrix* matrix_column_sum(const Matrix* mat) {
    int rows = mat->row;
    int cols = mat->col;

    Matrix* result = matrix_create(rows, 1);
    if (!result) return NULL;

    for (int i = 0; i < rows; i++) {
        double sum = 0.0;
        for (int j = 0; j < cols; j++) {
            sum += mat->data[i][j];
        }
        result->data[i][0] = sum;
    }

    return result;
}

void matrix_copy(Matrix* m1, Matrix* m2) {
    if(m1 == NULL || m2 == NULL || m1->row != m2->row || m1->col != m2->col) {
        printf("Unable to copy matrix");  
        return;
    }

    for (int r = 0; r < m1->row; r++) {
        for (int c = 0; c < m1->col; c++) {
            m2->data[r][c] = m1->data[r][c];
        }
    }
}

void matrix_print(Matrix* data) {
    for(int r = 0; r < data->row; r++){
        for(int c = 0; c < data->col; c++){
            printf("%f ", data->data[r][c]);
        }
        
        printf("\n");
    }
}

void matrix_free(Matrix* mat) {
    if (mat != NULL && mat->data != NULL) {
        // Libera ogni riga della matrice
        for (int i = 0; i < mat->row; i++) {
            if (mat->data[i] != NULL) {
                free(mat->data[i]);
                mat->data[i] = NULL;
            }
        }
        free(mat->data);
        mat->data = NULL;
    }
    free(mat);
}