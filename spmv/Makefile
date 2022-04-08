all: spmv

#CUDA_PARAMETERS
CUDA_PATH = /usr/local/cuda-11.2
ARCH = 70
CUDA_CC = ${CUDA_PATH}/bin/nvcc
NVCC_FLAGS = -O3 -w -m64 -gencode=arch=compute_$(ARCH),code=sm_$(ARCH) -gencode=arch=compute_$(ARCH),code=compute_$(ARCH)
CUDA_INCLUDES = -I$(CUDA_PATH)/include -I${CUDA_PATH}/samples/common/inc
CUDA_LIBS = -L$(CUDA_PATH)/lib64 -lcudart -lcusparse

CUSP_INCLUDES = -I/home/GaoJH/spmv/cusplibrary

spmv.o: spmv.cu
	$(CUDA_CC) $(NVCC_FLAGS) -o spmv.o -c spmv.cu $(CUSP_INCLUDES) $(CUDA_INCLUDES) 
main.o: main.cpp
	$(CUDA_CC) -ccbin g++ $(NVCC_FLAGS) -o main.o -c main.cpp $(CUSP_INCLUDES) $(CUDA_INCLUDES) 
spmv: spmv.o main.o
	$(CUDA_CC) $(NVCC_FLAGS) spmv.o main.o -o spmv $(CUSP_INCLUDES) $(CUDA_INCLUDES) $(CUDA_LIBS)

clean:
	rm *.o 
