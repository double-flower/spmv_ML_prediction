#pragma once

#include <cuda_runtime.h>
#include "cusparse.h"

// #define VALUE_TYPE float

#define NUM_RUN 1
#define NUM_TRANSFER 1
#define T_LONG_ROW 1024
#define T_SHORT_ROW 2.5
#define T_GAP 25
#define T_MIN_GAP 1
#define WINDOW_SIZE 16
#define T_SEGMENT 1024  // threshold of segments
#define N_SEGMENT 32 // number of tested segments 
#define NUM_BLOCKS_NEW 256
#define THREADS_PER_BLOCK_NEW 512

typedef unsigned int IndexType;
typedef double VALUE_TYPE;

int readMtx(char *filename, int &m, int &n, int &nnzA, IndexType *&csrRowPtrA, IndexType *&csrColIdxA,
	VALUE_TYPE *&csrValA);
void queryDevice();
inline void checkcuda(cudaError_t result);
inline void checkcusparse(cusparseStatus_t result);
template <unsigned int THREADS_PER_VECTOR>
void CSR_vector_spmv_prepare1(int M, IndexType *RowPtr, IndexType *ColIdx, VALUE_TYPE *Val, 
    VALUE_TYPE *x, VALUE_TYPE *y);
void CSR_vector_spmv_prepare0(int M, int TPV, IndexType *RowPtr, IndexType *ColIdx, 
    VALUE_TYPE *Val, VALUE_TYPE *x, VALUE_TYPE *y);
void generalSpMV_prepare(int M, int TPV, unsigned int sign, int n_segments, IndexType* row_segment, 
    IndexType* numThreadBlocks, IndexType* TPV_segment, IndexType *d_csrRowPtrA, IndexType *d_csrColIdxA, 
    VALUE_TYPE *d_csrValA, VALUE_TYPE *x_device, VALUE_TYPE *y_device);
void generalSpMV(int M, int N, int nnz, IndexType *csrRow, IndexType *csrCol, VALUE_TYPE *csrVal, 
    VALUE_TYPE *xHostPtr, VALUE_TYPE *yHostPtr);
void segment_matrix(int M, int nnz, IndexType *csrRow, int &n_segments, IndexType *row_segment, 
    IndexType *nnz_segment, IndexType *TPV_segment, unsigned int &sign);
int getTPV(int ave_nnz);
void calMtxFeatures(int M, int N, int nnz, IndexType* csrRow);
void calSegmentFeatures(int M, int N, int n_segments, unsigned int sign, 
    IndexType* row_segment, IndexType* nnz_segment, IndexType* csrRow);