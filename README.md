# spmv_ML_prediction
A machine learning based SpMV optimization approach

## Prerequisites
- **OS**: Ubuntu 18.04
- **CUDA**: 10.2 and later
- **[CUSP](https://github.com/cusplibrary/cusplibrary)**: 0.5.1 

## spmv
code for sparse matrix-vector multiplication
### Usage
- Modify Makefile
  - change `CUDA_PATH` to your own cuda path
  - change `CUSP_INCLUDES` to your own cusplibrary path
- compile: `make`
- run: `./spmv matrixfile`

## AutoML
code for training and testing models using AutoML tool
