#
# The following must be provided:
#

# A path to CUDA top directory
CUDA_TOP_DIR	:= ..
# An executable name
EXECUTABLE	:= bfs
# Cuda source files (compiled with nvcc)
CUFILES		:= bfs_cuda_kernels.cu
# C/C++ source files (compiled with gcc / c++)
CCFILES		:= bfs_cuda.c

# init C flags (more options will be added)
CFLAGS		:= -lm

include ../util/nvidia.mk
