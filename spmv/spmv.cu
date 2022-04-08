#include <cuda_runtime.h>
#include <stdio.h>
#include <string>
#include <stdlib.h> 
#include <iostream>
#include <fstream>
#include <ctype.h>
#include <string.h>
#include <math.h>
#include <cfloat>
#include <errno.h>
#include "cusparse.h"
#include "mmio.h"
#include <dirent.h>
#include <sys/time.h>
#include "spmv.h" 
#include <cusp/system/cuda/arch.h>
#include "cuda_profiler_api.h"
#include "my_timer.h"
  
using namespace std;

extern char matrixName[];

inline void checkcuda(cudaError_t result)
{
	if (result != cudaSuccess)
	{
		printf("CUDA Runtime Error: %s\n", cudaGetErrorString(result));
		printf("hello");
	}
}

inline void checkcusparse(cusparseStatus_t result)
{
	if(result != CUSPARSE_STATUS_SUCCESS){
		printf("CUSPARSE Error, error_code =  %d\n", result);
	}
}

int readMtx(char *filename, int &m, int &n, int &nnzA, IndexType *&csrRowPtrA, IndexType *&csrColIdxA,
	VALUE_TYPE *&csrValA)
{
	int ret_code = 0;
	MM_typecode matcode;

	FILE *f = NULL;
	int nnzA_mtx_report = 0;
	int isInteger = 0, isReal = 0, isPattern = 0, isSymmetric = 0;
	// load matrix
	if ((f = fopen(filename, "r")) == NULL)
		return -1;

	if (mm_read_banner(f, &matcode) != 0) {
		printf("Could not process Matrix Market banner.\n");
		return -2;
	}

	if (mm_is_complex(matcode)) {
		printf("Sorry, data type 'COMPLEX' is not supported. \n");
		return -3;
	}

	if (mm_is_pattern(matcode)) {
		isPattern = 1; printf("type = Pattern.\n");
	}

	if (mm_is_real(matcode)) {
		isReal = 1; printf("type = real.\n");
	}

	if (mm_is_integer(matcode)) {
		isInteger = 1; printf("type = integer.\n");
	}

	ret_code = mm_read_mtx_crd_size(f, &m, &n, &nnzA_mtx_report);
	if (ret_code != 0)
		return -4;

	if (mm_is_symmetric(matcode) || mm_is_hermitian(matcode)) {
		isSymmetric = 1;
		printf("symmetric = true.\n");
	}
	else {
		printf("symmetric = false.\n");
	}

	IndexType *csrRowPtrA_counter = (IndexType *)malloc((m + 1) * sizeof(IndexType));
	memset(csrRowPtrA_counter, 0, (m + 1) * sizeof(IndexType));

	IndexType *csrRowIdxA_tmp = (IndexType *)malloc(nnzA_mtx_report * sizeof(IndexType));
	memset(csrRowIdxA_tmp, 0, nnzA_mtx_report * sizeof(IndexType));
	IndexType *csrColIdxA_tmp = (IndexType *)malloc(nnzA_mtx_report * sizeof(IndexType));
	memset(csrColIdxA_tmp, 0, nnzA_mtx_report * sizeof(IndexType));
	VALUE_TYPE *csrValA_tmp = (VALUE_TYPE *)malloc(nnzA_mtx_report * sizeof(VALUE_TYPE));
	memset(csrValA_tmp, 0.0, nnzA_mtx_report * sizeof(VALUE_TYPE));

	for (int i = 0; i < nnzA_mtx_report; i++)
	{
		IndexType idxi = 0, idxj = 0;
		double fval = 0.0;
		int ival = 0;

		if (isReal)
			fscanf(f, "%d %d %lg\n", &idxi, &idxj, &fval);
		else if (isInteger)
		{
			fscanf(f, "%d %d %d\n", &idxi, &idxj, &ival);
			fval = ival;
		}
		else if (isPattern)
		{
			fscanf(f, "%d %d\n", &idxi, &idxj);
			fval = 1.0;
		}

		// adjust from 1-based to 0-based
		idxi--;
		idxj--;

		csrRowPtrA_counter[idxi]++;
		csrRowIdxA_tmp[i] = idxi;
		csrColIdxA_tmp[i] = idxj;
		csrValA_tmp[i] = fval;
	}

	if (f != stdin)
		fclose(f);	

	if (isSymmetric)
	{
		for (int i = 0; i < nnzA_mtx_report; i++) {
			if (csrRowIdxA_tmp[i] != csrColIdxA_tmp[i])
				csrRowPtrA_counter[csrColIdxA_tmp[i]]++;
		}
	}

	// exclusive scan for csrRowPtrA_counter
	IndexType old_val = 0, new_val = 0;

	old_val = csrRowPtrA_counter[0];
	csrRowPtrA_counter[0] = 0;
	for (int i = 1; i <= m; i++)
	{
		new_val = csrRowPtrA_counter[i];
		csrRowPtrA_counter[i] = old_val + csrRowPtrA_counter[i - 1];
		old_val = new_val;
	}

	nnzA = csrRowPtrA_counter[m];
	csrRowPtrA = (IndexType *)malloc((m + 1) * sizeof(IndexType));
	memcpy(csrRowPtrA, csrRowPtrA_counter, (m + 1) * sizeof(IndexType));
	memset(csrRowPtrA_counter, 0, (m + 1) * sizeof(IndexType));

	csrColIdxA = (IndexType *)malloc(nnzA * sizeof(IndexType));
	memset(csrColIdxA, 0, nnzA * sizeof(IndexType));
	csrValA = (VALUE_TYPE *)malloc(nnzA * sizeof(VALUE_TYPE));
	memset(csrValA, 0, nnzA * sizeof(VALUE_TYPE));

	if (isSymmetric)
	{
		for (int i = 0; i < nnzA_mtx_report; i++)
		{
			if (csrRowIdxA_tmp[i] != csrColIdxA_tmp[i])
			{
				int offset = csrRowPtrA[csrRowIdxA_tmp[i]] + csrRowPtrA_counter[csrRowIdxA_tmp[i]];
				csrColIdxA[offset] = csrColIdxA_tmp[i];
				csrValA[offset] = csrValA_tmp[i];
				csrRowPtrA_counter[csrRowIdxA_tmp[i]]++;

				offset = csrRowPtrA[csrColIdxA_tmp[i]] + csrRowPtrA_counter[csrColIdxA_tmp[i]];
				csrColIdxA[offset] = csrRowIdxA_tmp[i];
				csrValA[offset] = csrValA_tmp[i];
				csrRowPtrA_counter[csrColIdxA_tmp[i]]++;
			}
			else
			{
				int offset = csrRowPtrA[csrRowIdxA_tmp[i]] + csrRowPtrA_counter[csrRowIdxA_tmp[i]];
				csrColIdxA[offset] = csrColIdxA_tmp[i];
				csrValA[offset] = csrValA_tmp[i];
				csrRowPtrA_counter[csrRowIdxA_tmp[i]]++;
			}
		}
	}
	else
	{
		for (int i = 0; i < nnzA_mtx_report; i++)
		{
			int offset = csrRowPtrA[csrRowIdxA_tmp[i]] + csrRowPtrA_counter[csrRowIdxA_tmp[i]];
			csrColIdxA[offset] = csrColIdxA_tmp[i];
			csrValA[offset] = csrValA_tmp[i];
			csrRowPtrA_counter[csrRowIdxA_tmp[i]]++;
		}
	}

	// free tmp space
	free(csrColIdxA_tmp);
	free(csrValA_tmp);
	free(csrRowIdxA_tmp);
	free(csrRowPtrA_counter);
	return 0;
}

template <  unsigned int THREADS_PER_VECTOR,
            unsigned int VECTORS_PER_BLOCK>
__global__ void CSR_vector_spmv_kernel(
    const int rows,
    const IndexType *Ap,
    const IndexType *Aj,
    const VALUE_TYPE *Ax,
    const VALUE_TYPE *x,
    VALUE_TYPE *y)
{
    __shared__ volatile VALUE_TYPE sdata[VECTORS_PER_BLOCK * THREADS_PER_VECTOR + THREADS_PER_VECTOR/2];  // padded to avoid reduction conditionals
    __shared__ volatile IndexType ptrs[VECTORS_PER_BLOCK][2];

    const int thread_id   = blockDim.x * blockIdx.x + threadIdx.x;    // global thread index
    const int thread_lane = threadIdx.x & (THREADS_PER_VECTOR - 1);          // thread index within the vector
    const int vector_id   = thread_id   /  THREADS_PER_VECTOR;               // global vector index
    const int vector_lane = threadIdx.x /  THREADS_PER_VECTOR;               // vector index within the block
    const int num_vectors = VECTORS_PER_BLOCK * gridDim.x;                   // total number of active vectors
 
    for(int row = vector_id; row < rows; row += num_vectors)
    // for(int row = 30098 + vector_id; row < 60098; row += num_vectors)
    {
        // use two threads to fetch Ap[row] and Ap[row+1]
        // this is considerably faster than the straightforward version
            if(thread_lane < 2)
                ptrs[vector_lane][thread_lane] = Ap[row + thread_lane];
        
		__syncwarp();
        const IndexType element_start    = ptrs[vector_lane][0];                   //same as: element_start = Ap[row];
        const IndexType element_stop      = ptrs[vector_lane][1];                   //same as: element_stop   = Ap[row+1];

        VALUE_TYPE sum = 0.0;

        if (THREADS_PER_VECTOR == 32 && element_stop - element_start > 32)
        {
           // ensure aligned memory access to Aj and Ax
           int jj = element_start - (element_start & (THREADS_PER_VECTOR - 1)) + thread_lane;

           // accumulate local sums
            if(jj >= element_start && jj < element_stop)
                sum += Ax[jj] * x[Aj[jj]];

            // accumulate local sums
            for(jj += THREADS_PER_VECTOR; jj < element_stop; jj += THREADS_PER_VECTOR) 
                sum += Ax[jj] * x[Aj[jj]];
        }
        else
        {
            // accumulate local sums
            for(int jj = element_start + thread_lane; jj < element_stop; jj += THREADS_PER_VECTOR) 
                sum += Ax[jj] * x[Aj[jj]];
        }

        sdata[threadIdx.x] = sum;
        VALUE_TYPE temp;

        for (int stride = THREADS_PER_VECTOR/2; stride>0; stride/=2)
        {
			__syncwarp();
            temp = sdata[threadIdx.x + stride];
            sdata[threadIdx.x] = sum = sum + temp;
        }
		__syncwarp();
        if (thread_lane == 0)  
        {
            y[row] = sdata[threadIdx.x];
        }
    }
}

template <unsigned int THREADS_PER_VECTOR>
void CSR_vector_spmv_prepare1(int M, IndexType *RowPtr, IndexType *ColIdx, VALUE_TYPE *Val, VALUE_TYPE *x, VALUE_TYPE *y)
{
    const int THREADS_PER_BLOCK = 256;
    const int VECTORS_PER_BLOCK = THREADS_PER_BLOCK / THREADS_PER_VECTOR;
	const int MAX_BLOCKS = cusp::system::cuda::detail::max_active_blocks(CSR_vector_spmv_kernel<THREADS_PER_VECTOR, VECTORS_PER_BLOCK>, THREADS_PER_BLOCK, 0);
	const int REQUIRED_BLOCKS = (M + VECTORS_PER_BLOCK -1)/VECTORS_PER_BLOCK;
	const int NUM_BLOCKS = std::min<int>(MAX_BLOCKS, REQUIRED_BLOCKS);
    CSR_vector_spmv_kernel<THREADS_PER_VECTOR, VECTORS_PER_BLOCK><<<NUM_BLOCKS, THREADS_PER_BLOCK, 0, 0>>>(
            M, RowPtr, ColIdx, Val, x, y);
}

void CSR_vector_spmv_prepare0(int M, int TPV, IndexType *RowPtr, IndexType *ColIdx, VALUE_TYPE *Val, VALUE_TYPE *x, VALUE_TYPE *y)
{
    if (TPV <= 2)
    {
        CSR_vector_spmv_prepare1<2>(M, RowPtr, ColIdx, Val, x, y);
        return;
    }
    else if (TPV <= 4)
    {
        CSR_vector_spmv_prepare1<4>(M, RowPtr, ColIdx, Val, x, y);
        return;
    }
    else if (TPV <= 8)
    {
        CSR_vector_spmv_prepare1<8>(M, RowPtr, ColIdx, Val, x, y);
        return;
    }
    else if (TPV <= 16)
    {
        CSR_vector_spmv_prepare1<16>(M, RowPtr, ColIdx, Val, x, y);
        return;
    }

    CSR_vector_spmv_prepare1<32>(M, RowPtr, ColIdx, Val, x, y);
}

// two blocks: the first is a short-row block, and the second is a long-row block
__global__ void CSR_block_spmv_kernel_2_10(
    const int rows,
    const int row_segment_0,
    const int row_segment_1,
    const int num_blocks_0, 
    const int num_blocks_1,
    const int TPV_0,
    const IndexType *Ap,
    const IndexType *Aj,
    const VALUE_TYPE *Ax,
    const VALUE_TYPE *x,
    VALUE_TYPE *y)
{
    __shared__ volatile VALUE_TYPE sdata[T_LONG_ROW];
    extern __shared__ IndexType ptrs[][2];

    if (blockIdx.x < num_blocks_1) { // the first segment
        const int thread_id   = blockDim.x * blockIdx.x + threadIdx.x;    // global thread index
        const int thread_lane = threadIdx.x & (TPV_0 - 1);          // thread index within the vector
        const int vector_id   = thread_id   /  TPV_0;               // global vector index
        const int vector_lane = threadIdx.x /  TPV_0;               // vector index within the block
        const int num_vectors = (blockDim.x / TPV_0) * num_blocks_1;                   // total number of active vectors
        for(int row = row_segment_0 + vector_id; row < row_segment_1; row += num_vectors)
        {
            if(thread_lane < 2)
                ptrs[vector_lane][thread_lane] = Ap[row + thread_lane];
            
            __syncwarp();
            const IndexType element_start    = ptrs[vector_lane][0];                   //same as: element_start = Ap[row];
            const IndexType element_stop      = ptrs[vector_lane][1];                   //same as: element_stop   = Ap[row+1];

            VALUE_TYPE sum = 0.0;
    
            if (TPV_0 == 32 && element_stop - element_start > 32)
            {
                // ensure aligned memory access to Aj and Ax
                int jj = element_start - (element_start & (TPV_0 - 1)) + thread_lane;
    
                // accumulate local sums
                if(jj >= element_start && jj < element_stop)
                    sum += Ax[jj] * x[Aj[jj]];
    
                // accumulate local sums
                for(jj += TPV_0; jj < element_stop; jj += TPV_0) 
                    sum += Ax[jj] * x[Aj[jj]];
            }
            else
            {
                // accumulate local sums
                for(int jj = element_start + thread_lane; jj < element_stop; jj += TPV_0) 
                    sum += Ax[jj] * x[Aj[jj]];
            }
    
            sdata[threadIdx.x] = sum;
            VALUE_TYPE temp;
    
            for (int stride = TPV_0/2; stride>0; stride/=2)
            {
                __syncwarp();
                temp = sdata[threadIdx.x + stride];
                sdata[threadIdx.x] = sum = sum + temp;
            }
            __syncwarp();
            if (thread_lane == 0)  
            {
                y[row] = sdata[threadIdx.x];
            }
        }
    } else { // the second segment
        IndexType vector_lane = (blockIdx.x - num_blocks_1);
        IndexType thread_lane = threadIdx.x;
        for (int row = row_segment_1 + vector_lane; row < rows; row += (gridDim.x - num_blocks_1)) {
            if (threadIdx.x < 2)
                ptrs[vector_lane][thread_lane] = Ap[row + thread_lane];
            
            __syncthreads();
            const IndexType element_start = ptrs[vector_lane][0];
            const IndexType element_stop = ptrs[vector_lane][1];
    
            VALUE_TYPE sum = 0.0;
            for (int i = element_start + thread_lane; i < element_stop; i += blockDim.x)
                sum += Ax[i] * x[Aj[i]];
            
            sdata[thread_lane] = sum;
            for (int stride = blockDim.x/2; stride > 0; stride /= 2) {
                __syncthreads();
                if (threadIdx.x < stride)
                    sdata[threadIdx.x] += sdata[threadIdx.x + stride];
            }
            __syncwarp();
            if (threadIdx.x == 0)
                y[row] = sdata[threadIdx.x];
        }
    }
}

// two blocks: the first is a long-row block, and the second is a short-row block
__global__ void CSR_block_spmv_kernel_2_01(
    const int rows,
    int row_segment_0,
    int row_segment_1,
    int num_blocks_0, 
    int num_blocks_1,
    int TPV_1,
    const IndexType *Ap,
    const IndexType *Aj,
    const VALUE_TYPE *Ax,
    const VALUE_TYPE *x,
    VALUE_TYPE *y)
{
    __shared__ volatile VALUE_TYPE sdata[T_LONG_ROW]; 
    extern __shared__ IndexType ptrs[][2];

    // the first segment
    if (blockIdx.x < num_blocks_1) {
        IndexType vector_lane = blockIdx.x;
        IndexType thread_lane = threadIdx.x;
        for (int row = row_segment_0 + vector_lane; row < row_segment_1; row += num_blocks_1) {
            if (threadIdx.x < 2)
                ptrs[vector_lane][thread_lane] = Ap[row + thread_lane];
            
            __syncthreads();
            const IndexType element_start = ptrs[vector_lane][0];
            const IndexType element_stop = ptrs[vector_lane][1];
    
            VALUE_TYPE sum = 0.0;
            for (int i = element_start + thread_lane; i < element_stop; i += blockDim.x)
                sum += Ax[i] * x[Aj[i]];
            
            sdata[thread_lane] = sum;
            for (int stride = blockDim.x/2; stride > 0; stride /= 2) {
                __syncthreads();
                if (threadIdx.x < stride)
                    sdata[threadIdx.x] += sdata[threadIdx.x + stride];
            }
            __syncwarp();
            if (threadIdx.x == 0)
                y[row] = sdata[threadIdx.x];
        }
    
    } else { // the second segment
        const int thread_id   = blockDim.x * (blockIdx.x - num_blocks_1) + threadIdx.x;    // global thread index
        const int thread_lane = threadIdx.x & (TPV_1 - 1);          // thread index within the vector
        const int vector_id   = thread_id   /  TPV_1;               // global vector index
        const int vector_lane = threadIdx.x /  TPV_1;               // vector index within the block
        const int num_vectors = (blockDim.x / TPV_1) * (gridDim.x - num_blocks_1);                   // total number of active vectors
        for(int row = row_segment_1 + vector_id; row < rows; row += num_vectors)
        {
            if(thread_lane < 2)
                ptrs[vector_lane][thread_lane] = Ap[row + thread_lane];
            
            __syncwarp();
            const IndexType element_start    = ptrs[vector_lane][0];                   //same as: element_start = Ap[row];
            const IndexType element_stop      = ptrs[vector_lane][1];                   //same as: element_stop   = Ap[row+1];

            VALUE_TYPE sum = 0.0;
    
            if (TPV_1 == 32 && element_stop - element_start > 32)
            {
                // ensure aligned memory access to Aj and Ax
                int jj = element_start - (element_start & (TPV_1 - 1)) + thread_lane;
    
                // accumulate local sums
                if(jj >= element_start && jj < element_stop)
                    sum += Ax[jj] * x[Aj[jj]];
    
                // accumulate local sums
                for(jj += TPV_1; jj < element_stop; jj += TPV_1) 
                    sum += Ax[jj] * x[Aj[jj]];
            }
            else
            {
                // accumulate local sums
                for(int jj = element_start + thread_lane; jj < element_stop; jj += TPV_1) 
                    sum += Ax[jj] * x[Aj[jj]];
            }
    
            sdata[threadIdx.x] = sum;
            VALUE_TYPE temp;
    
            for (int stride = TPV_1/2; stride>0; stride/=2)
            {
                __syncwarp();
                temp = sdata[threadIdx.x + stride];
                sdata[threadIdx.x] = sum = sum + temp;
            }
            __syncwarp();
            if (thread_lane == 0)  
            {
                y[row] = sdata[threadIdx.x];
            }
        }
    }
}

// three blocks: the first and third are short-row blocks, and the second is a long-row block
__global__ void CSR_block_spmv_kernel_3(
    const int rows,
    const int row_segment_0,
    const int row_segment_1,
    const int row_segment_2,
    const int num_blocks_0, 
    const int num_blocks_1,
    const int num_blocks_2,
    const int TPV_0,
    const int TPV_2,
    const IndexType *Ap,
    const IndexType *Aj,
    const VALUE_TYPE *Ax,
    const VALUE_TYPE *x,
    VALUE_TYPE *y)
{
    __shared__ volatile VALUE_TYPE sdata[T_LONG_ROW];  // padded to avoid reduction conditionals
    extern __shared__ IndexType ptrs[][2];
    
    if (blockIdx.x < num_blocks_1) { // first block: short-row block
        const int thread_id   = blockDim.x * blockIdx.x + threadIdx.x;    // global thread index
        const int thread_lane = threadIdx.x & (TPV_0 - 1);          // thread index within the vector
        const int vector_id   = thread_id   /  TPV_0;               // global vector index
        const int vector_lane = threadIdx.x /  TPV_0;               // vector index within the block
        const int num_vectors = (blockDim.x / TPV_0) * num_blocks_1;                   // total number of active vectors
        for(int row = row_segment_0 + vector_id; row < row_segment_1; row += num_vectors)
        {
            if(thread_lane < 2)
                ptrs[vector_lane][thread_lane] = Ap[row + thread_lane];
            
            __syncwarp();
            const IndexType element_start    = ptrs[vector_lane][0];                   //same as: element_start = Ap[row];
            const IndexType element_stop      = ptrs[vector_lane][1];                   //same as: element_stop   = Ap[row+1];

            VALUE_TYPE sum = 0.0;
    
            if (TPV_0 == 32 && element_stop - element_start > 32)
            {
                // ensure aligned memory access to Aj and Ax
                int jj = element_start - (element_start & (TPV_0 - 1)) + thread_lane;
    
                // accumulate local sums
                if(jj >= element_start && jj < element_stop)
                    sum += Ax[jj] * x[Aj[jj]];
    
                // accumulate local sums
                for(jj += TPV_0; jj < element_stop; jj += TPV_0) 
                    sum += Ax[jj] * x[Aj[jj]];
            }
            else
            {
                // accumulate local sums
                for(int jj = element_start + thread_lane; jj < element_stop; jj += TPV_0) 
                    sum += Ax[jj] * x[Aj[jj]];
            }
    
            sdata[threadIdx.x] = sum;
            VALUE_TYPE temp;
    
            for (int stride = TPV_0/2; stride>0; stride/=2)
            {
                __syncwarp();
                temp = sdata[threadIdx.x + stride];
                sdata[threadIdx.x] = sum = sum + temp;
            }
            __syncwarp();
            if (thread_lane == 0)  
            {
                y[row] = sdata[threadIdx.x];
            }
        }
    } else if (blockIdx.x < num_blocks_2) { // second block: long-row block
        IndexType vector_lane = (blockIdx.x - num_blocks_1);
        IndexType thread_lane = threadIdx.x;
        for (int row = row_segment_1 + vector_lane; row < row_segment_2; row += (num_blocks_2 - num_blocks_1)) {
            if (threadIdx.x < 2)
                ptrs[vector_lane][thread_lane] = Ap[row + thread_lane];
            
            __syncthreads();
            const IndexType element_start = ptrs[vector_lane][0];
            const IndexType element_stop = ptrs[vector_lane][1];
    
            VALUE_TYPE sum = 0.0;
            for (int i = element_start + thread_lane; i < element_stop; i += blockDim.x)
                sum += Ax[i] * x[Aj[i]];
            
            sdata[thread_lane] = sum;
            for (int stride = blockDim.x/2; stride > 0; stride /= 2) {
                __syncthreads();
                if (threadIdx.x < stride)
                    sdata[threadIdx.x] += sdata[threadIdx.x + stride];
            }
            __syncwarp();
            if (threadIdx.x == 0)
                y[row] = sdata[threadIdx.x];
        }
    } else { // third block: short-row block           
        const int thread_id   = blockDim.x * (blockIdx.x - num_blocks_2) + threadIdx.x;    // global thread index
        const int thread_lane = threadIdx.x & (TPV_2 - 1);          // thread index within the vector
        const int vector_id   = thread_id   /  TPV_2;               // global vector index
        const int vector_lane = threadIdx.x /  TPV_2;               // vector index within the block
        const int num_vectors = (blockDim.x / TPV_2) * (gridDim.x - num_blocks_2);                   // total number of active vectors
        for(int row = row_segment_2 + vector_id; row < rows; row += num_vectors)
        {
            if(thread_lane < 2)
                ptrs[vector_lane][thread_lane] = Ap[row + thread_lane];
            
            __syncwarp();
            const IndexType element_start    = ptrs[vector_lane][0];                   //same as: element_start = Ap[row];
            const IndexType element_stop      = ptrs[vector_lane][1];                   //same as: element_stop   = Ap[row+1];

            VALUE_TYPE sum = 0.0;
    
            if (TPV_2 == 32 && element_stop - element_start > 32)
            {
                    // ensure aligned memory access to Aj and Ax
                    int jj = element_start - (element_start & (TPV_2 - 1)) + thread_lane;
        
                    // accumulate local sums
                    if(jj >= element_start && jj < element_stop)
                        sum += Ax[jj] * x[Aj[jj]];
        
                    // accumulate local sums
                    for(jj += TPV_2; jj < element_stop; jj += TPV_2) 
                        sum += Ax[jj] * x[Aj[jj]];
            }
            else
            {
                // accumulate local sums
                for(int jj = element_start + thread_lane; jj < element_stop; jj += TPV_2) 
                    sum += Ax[jj] * x[Aj[jj]];
            }
    
            sdata[threadIdx.x] = sum;
            VALUE_TYPE temp;
    
            for (int stride = TPV_2/2; stride>0; stride/=2)
            {
                __syncwarp();
                temp = sdata[threadIdx.x + stride];
                sdata[threadIdx.x] = sum = sum + temp;
            }
            __syncwarp();
            if (thread_lane == 0)  
            {
                y[row] = sdata[threadIdx.x];
            }
        }
    } 
}

void generalSpMV_prepare(int M, int TPV, unsigned int sign, int n_segments, IndexType* row_segment, 
    IndexType* numThreadBlocks, IndexType* TPV_segment, IndexType *d_csrRowPtrA, IndexType *d_csrColIdxA, 
    VALUE_TYPE *d_csrValA, VALUE_TYPE *x_device, VALUE_TYPE *y_device)
{
    int shared_mem_size;
    if (n_segments == 1 || n_segments > 3) {
        CSR_vector_spmv_prepare0(M, TPV, d_csrRowPtrA, d_csrColIdxA, d_csrValA, x_device,y_device);
    } else if (n_segments == 2){
        int min_tmp = 0, num_long_rows = 0;
        if (sign == 1) {
            num_long_rows = row_segment[1] - row_segment[0];
            min_tmp = TPV_segment[1];
            min_tmp = THREADS_PER_BLOCK_NEW / min_tmp;
            min_tmp = (num_long_rows > min_tmp)? num_long_rows:min_tmp;
            shared_mem_size = min_tmp*2*sizeof(IndexType);
            CSR_block_spmv_kernel_2_01<<<NUM_BLOCKS_NEW, THREADS_PER_BLOCK_NEW, shared_mem_size, 0>>>
                (M, row_segment[0], row_segment[1], numThreadBlocks[0], numThreadBlocks[1],
                    TPV_segment[1], d_csrRowPtrA, d_csrColIdxA, d_csrValA, x_device, y_device);
        } else {
            num_long_rows = row_segment[2] - row_segment[1];
            min_tmp = TPV_segment[0];
            min_tmp = THREADS_PER_BLOCK_NEW / min_tmp;
            min_tmp = (num_long_rows > min_tmp)? num_long_rows:min_tmp;
            shared_mem_size = min_tmp*2*sizeof(IndexType);
            CSR_block_spmv_kernel_2_10<<<NUM_BLOCKS_NEW, THREADS_PER_BLOCK_NEW, shared_mem_size, 0>>>
                (M, row_segment[0], row_segment[1], numThreadBlocks[0], numThreadBlocks[1],
                    TPV_segment[0], d_csrRowPtrA, d_csrColIdxA, d_csrValA, x_device, y_device);
        }
    } else { // n_segments == 3
        int min_tmp = (TPV_segment[0] < TPV_segment[2])? TPV_segment[0]: TPV_segment[2];
        min_tmp = THREADS_PER_BLOCK_NEW / min_tmp;
        int num_long_rows = row_segment[2] - row_segment[1];
        min_tmp = (num_long_rows > min_tmp)? num_long_rows:min_tmp;
        shared_mem_size = min_tmp*2*sizeof(IndexType);
        CSR_block_spmv_kernel_3<<<NUM_BLOCKS_NEW, THREADS_PER_BLOCK_NEW, shared_mem_size, 0>>>
            (M, row_segment[0], row_segment[1], row_segment[2], 
                numThreadBlocks[0], numThreadBlocks[1], numThreadBlocks[2],
                TPV_segment[0], TPV_segment[2], 
                d_csrRowPtrA, d_csrColIdxA, d_csrValA, x_device, y_device);
    }
}

void segment_matrix(int M, int nnz, IndexType *csrRow, int &n_segments, IndexType *row_segment, IndexType *nnz_segment, 
        IndexType *TPV_segment, unsigned int &sign)
{
    int sum_nnz = 0;
    int last_is_long_row = 0;
    n_segments = 0;
    sign = 0;
    for (int i = 0; i < M; i++) {
        int cur_nnz = csrRow[i+1] - csrRow[i];
        sum_nnz += cur_nnz;
        if (cur_nnz >= T_LONG_ROW) { // long row
            if (i == 0) {
                last_is_long_row = 1;
                row_segment[n_segments++] = 0;
                continue;
            } else if (!last_is_long_row) {    // last row is short row, then blocking and recording border
                row_segment[n_segments] = i;
                last_is_long_row = 1;
                nnz_segment[n_segments-1] = sum_nnz - cur_nnz;
                TPV_segment[n_segments-1] = getTPV(nnz_segment[n_segments-1] / 
                    (row_segment[n_segments] - row_segment[n_segments-1]));
                sum_nnz = cur_nnz;
                n_segments++;
            }
        } else {    // short row
            if (i == 0) {
                row_segment[n_segments++] = 0;
                last_is_long_row = 0;
                continue;
            } else if (last_is_long_row) { // last row is long row
                row_segment[n_segments] = i;
                last_is_long_row = 0;
                nnz_segment[n_segments-1] = sum_nnz - cur_nnz;
                TPV_segment[n_segments-1] = THREADS_PER_BLOCK_NEW;
                sum_nnz = cur_nnz;
                sign += (int)pow(2, n_segments-1);
                n_segments++;
            }  
        }
    }
    row_segment[n_segments] = M;
    nnz_segment[n_segments-1] = nnz;
    for (int i = 0; i < n_segments-1; i++)
        nnz_segment[n_segments-1] -= nnz_segment[i];

    if (last_is_long_row) {
        sign += (int)pow(2, n_segments-1);
        TPV_segment[n_segments-1] = THREADS_PER_BLOCK_NEW;
    } else {
        TPV_segment[n_segments-1] = getTPV(nnz_segment[n_segments-1] / 
            (row_segment[n_segments] - row_segment[n_segments-1]));
    }

    // validate    
    IndexType sum = 0;
    for (int i = 0; i < n_segments; i++)  sum += nnz_segment[i];
    if (sum != nnz) printf("WARNING: nnz of some segments is wrong!!!\n");
}

int getTPV(int ave_nnz)
{
    if (ave_nnz <= 2)
        return 2;
    else if (ave_nnz <= 4)
        return 4;
    else if (ave_nnz <= 8)
        return 8;
    else if (ave_nnz <=16)
        return 16;
    else 
        return 32;
}

void calMtxFeatures(int M, int N, int nnz, IndexType* csrRow)
{
	cpu_timer timer;
	timer.start();

	IndexType max = 0, min = 2147483647;
	double mean = (double)nnz / (double)M;
	double variance = 0.0;
	for (int i = 0; i < M; i++)
	{
		IndexType nnz_row = csrRow[i+1] - csrRow[i];
		max = (nnz_row > max) ? nnz_row : max;
		min = (nnz_row < min) ? nnz_row : min;
		variance += (nnz_row - mean) * (nnz_row - mean);
	}
	variance /= (double)M;
	double density = (double)nnz / ((double)M * N);
	double cov= sqrt(variance) / mean;
	double max_mu = max - mean;
	double elapsed_time = timer.get();

    // FILE *fresult = fopen("mtx_features.txt", "a+");
 	// if (fresult != NULL) {
 	// 	char ch=fgetc(fresult);
 	// 	if (ch == EOF) {// file is empty 
 	// 		fprintf(fresult, "Matrix rows cols nnz min max mean variance density cov max_mu");
 	// 	}
 	// }
 	// else {
 	// 	printf("open file failed\n");
 	// }
    // fprintf(fresult, "%s %d %d %d %d %d %f %f %f %f %f\n", matrixName, M, N, nnz, min, max, mean, variance, density, cov, max_mu);
	// fclose(fresult);
}

void calSegmentFeatures(int M, int N, int n_segments, unsigned int sign, 
    IndexType* row_segment, IndexType* nnz_segment, IndexType* csrRow)
{
	cpu_timer timer;
	timer.start();

    for (int i = 0; i < n_segments; i++) {
        unsigned int temp_sign = sign % 2;
        if (temp_sign == 0) {
            int temp_M = row_segment[i+1] - row_segment[i];
            int temp_nnz = nnz_segment[i];
            IndexType max = 0, min = 2147483647;
            double mean = (double)temp_nnz / (double)temp_M;
            double variance = 0.0;
            for (int j = row_segment[i]; j < row_segment[i+1]; j++) {
                IndexType nnz_row = csrRow[j+1] - csrRow[j];
                max = (nnz_row > max) ? nnz_row : max;
                min = (nnz_row < min) ? nnz_row : min;
                variance += (nnz_row - mean) * (nnz_row - mean);
            }
            variance /= temp_M;
            double density = (double)temp_nnz / ((double)temp_M * N);
            double cov= sqrt(variance) / mean;
            double max_mu = max - mean;
            double s_mean = sqrt(mean);
        }
        sign = sign >> 1;
    }

	double elapsed_time = timer.get();
}

void generalSpMV(int M, int N, int nnz, IndexType *csrRow, IndexType *csrCol, VALUE_TYPE *csrVal, VALUE_TYPE *xHostPtr, VALUE_TYPE *yHostPtr)
{
    IndexType *d_csrRowPtrA, *d_csrColIdxA;
    VALUE_TYPE * d_csrValA;

    IndexType row_segment[T_SEGMENT+1] = {0};
    IndexType nnz_segment[T_SEGMENT] = {0};
    IndexType TPV_segment[T_SEGMENT] = {0};  // save TPV of short row blocks
    IndexType numThreadBlocks[T_SEGMENT+1] = {0};
    unsigned int sign = 0; // for example, sign=(10)_2=(1)_10 represents that this matrix includes two segment, and the first segment comprises of long rows
    int n_segments = 0;

    // blocking
    segment_matrix(M, nnz, csrRow, n_segments, row_segment, nnz_segment, TPV_segment, sign);
    
    // calculate matrix or block features
    if(n_segments == 1 || n_segments > 3) {
        calMtxFeatures(M, N, nnz, csrRow);
    } else {
        calSegmentFeatures(M, N, n_segments, sign, row_segment, nnz_segment, csrRow);
    }

    // calculate number of thread blocks assigned to each matrix block
    unsigned int temp_sign = sign;
    numThreadBlocks[0] = 0;
    if (n_segments == 2) {
        if (sign == 1) { // sign = (01)_2
            numThreadBlocks[1] = row_segment[1] - row_segment[0];
            numThreadBlocks[2] = NUM_BLOCKS_NEW;
        } else { // sign = (10)_2
            numThreadBlocks[1] = NUM_BLOCKS_NEW - (row_segment[2] - row_segment[1]);
            numThreadBlocks[2] = NUM_BLOCKS_NEW;
        }
    }
    if (n_segments == 3) {
        IndexType num_long_rows = row_segment[2] - row_segment[1];
        numThreadBlocks[1] = ((NUM_BLOCKS_NEW - num_long_rows) * nnz_segment[0]) /
                         (nnz - nnz_segment[1]);
        numThreadBlocks[1] = (numThreadBlocks[1] < 1)? 1:numThreadBlocks[1];
        numThreadBlocks[2] = numThreadBlocks[1] + num_long_rows;
        numThreadBlocks[3] = NUM_BLOCKS_NEW;
    }

    printf("\n\n========segments info start========\n");
    printf("number of segments: %d sign:%d\n", n_segments, sign);
    printf("segment-id\trow_start\trow_end\t\tnnz\t\tave_nnz\t\tsign\t\tnum_block\t\tTPV\n");
    temp_sign = sign;
    for (int i = 0; i < n_segments; i++) {
        unsigned int temp = temp_sign % 2;
        printf("segment-%2d:\t%d\t\t%d\t\t%d\t\t%d\t\t%d\t\t%d\t\t\t%d\n", i, row_segment[i], row_segment[i+1], nnz_segment[i], 
            nnz_segment[i]/(row_segment[i+1]-row_segment[i]), temp, numThreadBlocks[i+1]-numThreadBlocks[i], TPV_segment[i]);
        temp_sign /= 2;
    }
    printf("\n========segments info end========\n");
    
    // prepare matrix on GPU
    checkcuda(cudaMalloc((void **)&d_csrRowPtrA, (M+1) * sizeof(IndexType)));
	checkcuda(cudaMalloc((void **)&d_csrColIdxA, nnz * sizeof(IndexType)));
    checkcuda(cudaMalloc((void **)&d_csrValA, nnz * sizeof(VALUE_TYPE)));

	// prepare vector x and y
	VALUE_TYPE *y_device, *x_device;
    checkcuda(cudaMalloc((void**)&x_device, N * sizeof(VALUE_TYPE)));
    checkcuda(cudaMalloc((void**)&y_device, M * sizeof(VALUE_TYPE)));
    cudaMemset(y_device, M * sizeof(VALUE_TYPE), 0.0);

    // copy vector x
    checkcuda(cudaMemcpy(x_device, xHostPtr, N * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice));
    
    // transfer data from cpu to gpu
    checkcuda(cudaMemcpy(d_csrRowPtrA, csrRow, (M+1) * sizeof(IndexType), cudaMemcpyHostToDevice));
    checkcuda(cudaMemcpy(d_csrColIdxA, csrCol, nnz * sizeof(IndexType), cudaMemcpyHostToDevice));
    checkcuda(cudaMemcpy(d_csrValA, csrVal, nnz * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice));

    // TPV
    // warmp up
    generalSpMV_prepare(M, nnz/M, sign, n_segments, row_segment, numThreadBlocks, TPV_segment, 
        d_csrRowPtrA, d_csrColIdxA, d_csrValA, x_device, y_device);

    cuda_timer spmv_timer;
    spmv_timer.start();
    for (int i = 0; i < NUM_RUN; i++) {
        generalSpMV_prepare(M, nnz/M, sign, n_segments, row_segment, numThreadBlocks, TPV_segment, 
        d_csrRowPtrA, d_csrColIdxA, d_csrValA, x_device, y_device);
    }
    float spmv_time = spmv_timer.stop() / NUM_RUN;
 	
    checkcuda(cudaMemcpy(yHostPtr, y_device, sizeof(VALUE_TYPE) * M, cudaMemcpyDeviceToHost));
    
	// free resource
	checkcuda(cudaFree(x_device));
	checkcuda(cudaFree(y_device));
    checkcuda(cudaFree(d_csrRowPtrA));
	checkcuda(cudaFree(d_csrColIdxA));
    checkcuda(cudaFree(d_csrValA));
	cudaDeviceReset();
}