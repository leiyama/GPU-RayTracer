CC = /usr/bin/g++

LD_FLAGS = -lrt

# CUDA code generation flags
GENCODE_FLAGS   := -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35

# OS-specific build flags
ifneq ($(DARWIN),)
      CUDA_PATH ?= /usr/local/cuda
      LDFLAGS   := -Xlinker -rpath $(CUDA_LIB_PATH) -L$(CUDA_LIB_PATH) -lcudart
      CCFLAGS   := -arch $(OS_ARCH)
else
      CUDA_PATH ?= /usr/local/cuda
  ifeq ($(OS_SIZE),32)
      LDFLAGS   := -L$(CUDA_LIB_PATH) -lcudart
      CCFLAGS   := -m32
  else
      LDFLAGS   := -L$(CUDA_LIB_PATH) -lcudart
      CCFLAGS   := -m64
  endif
endif

CUDA_INC_PATH   ?= $(CUDA_PATH)/include
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin
CUDA_LIB_PATH   ?= $(CUDA_PATH)/lib64

# NVCC
NVCC            ?= $(CUDA_BIN_PATH)/nvcc

# OS-architecture specific flags
ifeq ($(OS_SIZE),32)
      NVCCFLAGS := -m32
else
      NVCCFLAGS := -m64
endif

all: raytrace

matrix.o: matrix.cu
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) $(EXTRA_NVCCFLAGS) -rdc=true -I$(CUDA_INC_PATH) -c -line_info matrix.cu

raytrace.o: raytrace_cuda.cu
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) $(EXTRA_NVCCFLAGS) -rdc=true -I$(CUDA_INC_PATH) -c -line_info raytrace_cuda.cu

camera.o: camera.cpp
	$(CC) -I$(CUDA_INC_PATH) -c camera.cpp 

parser.o: parser.cpp
	$(CC) -I$(CUDA_INC_PATH) -c parser.cpp 

point.o: point.cu
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) $(EXTRA_NVCCFLAGS) -rdc=true -I$(CUDA_INC_PATH) -c -line_info point.cu

superquadric.o: superquadric.cu
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) $(EXTRA_NVCCFLAGS) -rdc=true -I$(CUDA_INC_PATH) -c -line_info superquadric.cu 

testing.o: testing.cu
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) $(EXTRA_NVCCFLAGS) -rdc=true -I$(CUDA_INC_PATH) -c -line_info testing.cu 

raytrace: testing.o matrix.o superquadric.o point.o parser.o camera.o raytrace.o
	$(NVCC) $(GENCODE_FLAGS) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) -I$(CUDA_INC_PATH) -DTHRUST_DEBUG testing.o  superquadric.o matrix.o point.o parser.o camera.o raytrace_cuda.o -o raytracer

clean:
	rm -f *.o raytracer
