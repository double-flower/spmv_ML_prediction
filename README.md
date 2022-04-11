# Machine Learning based SpMV Optimization

## Overview
Replication package for sparse matrix-vector multiplication
 
 **/AutoML**: Source code for training and testing ML models using AutoML tool
 
 **/spmv**: Source code of our approach (SpMV algorithm)

## Requirements
- **OS**: Ubuntu 18.04
- **CUDA**: >= 10.2
- **[CUSP](https://github.com/cusplibrary/cusplibrary)**: 0.5.1 

## Usage
#### 1. Makefile Setup ####

Setting *CUDA_PATH* to local cuda path
  
  `CUDA_PATH = /usr/loca/cuda`
  
Setting *CUSP_INCLUDES* to local cusplibrary path
  
  `CUSP_INCLUDES = path2cusplibrary`
  
#### 2. Compile: ####

 `make`
 
 #### 3. Run: ####

`./spmv matrixfile`
