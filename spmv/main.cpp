#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <ctype.h>
#include <string.h>
#include <math.h>
#include <cfloat>
#include <errno.h>
#include "spmv.h"
#include <dirent.h>

using namespace std;

/*** Declaration ***/
int M, N, nnz;

IndexType *csrRowIndexHostPtr = 0;
IndexType *csrColIndexHostPtr = 0;
VALUE_TYPE *csrValHostPtr = 0;
VALUE_TYPE *xHostPtr = 0;
VALUE_TYPE *yHostPtr = 0;

char matrixName[1024] = {0};

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        printf("usage: ./exe MatrixName\n");
        return 0;
    }

    printf("%s %s\n", argv[0], argv[1]);

    char file_name[1024] = {0};
    char temp[1024] = {0};

    strcpy(file_name, argv[1]);

    strcpy(temp, argv[1]);
    char *mtx_pure_name = strrchr(temp, '/');
    int len = strlen(mtx_pure_name);
    mtx_pure_name[len - 4] = '\0';
    strcpy(matrixName, mtx_pure_name+1);

    printf("reading file %s\n", file_name);
    readMtx(file_name, M, N, nnz, csrRowIndexHostPtr, csrColIndexHostPtr, csrValHostPtr);
    printf("M=%d N=%d nnz=%d\n", M, N, nnz);

	// calMtxFeatures(M, N, nnz, csrRowIndexHostPtr, csrColIndexHostPtr, csrValHostPtr);

    xHostPtr = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * N);
    for (int i = 0; i < N; i++)
        xHostPtr[i] = 1.0;
    yHostPtr = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * M);
    memset(yHostPtr, 0.0, sizeof(VALUE_TYPE) * M);
    
    // Calculate accurate result
    VALUE_TYPE *y_ref = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * M);
    for (int i = 0; i < M; i++)
    {
        y_ref[i] = 0.0;
        for (IndexType j = csrRowIndexHostPtr[i]; j < csrRowIndexHostPtr[i + 1]; j++)
        {
            y_ref[i] += csrValHostPtr[j] * xHostPtr[csrColIndexHostPtr[j]];
        }
    }

    // call SpMV
    printf("\nBegain call spmv function.\n");
    generalSpMV(M, N, nnz, csrRowIndexHostPtr, csrColIndexHostPtr, csrValHostPtr, xHostPtr, yHostPtr);
    printf("End call spmv function.\n\n");

    // validate calculated result
    int counter_wrong = 0;
    for (int i = 0; i < M; i++)
    {
        if (abs(yHostPtr[i] - y_ref[i]) > 1e-6)
        {
            if (counter_wrong == 0)
                printf("yHostPtr[%d]=%f, y_ref[%d]=%f\n", i, yHostPtr[i], i, y_ref[i]);
            counter_wrong++;
        }
    }
    printf("Warning: %d are wrong!\n", counter_wrong);

    //free memory
    free(xHostPtr);
    free(yHostPtr);
    free(y_ref);
    free(csrRowIndexHostPtr);
    free(csrColIndexHostPtr);
    free(csrValHostPtr);

    return 0;
}