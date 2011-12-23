#
# The following must be provided:
#

# A path to CUDA top directory
CUDA_TOP_DIR	:= ..
# An executable name
EXECUTABLE	:= hotspot
# Cuda source files (compiled with nvcc)
CUFILES		:= hotspot_cuda_kernel.cu
# C/C++ source files (compiled with gcc / c++)
CCFILES		:= hotspot_cuda.c

# init C flags (more options will be added)
CFLAGS		:= -lm
# use g++ instead of gcc

include ../util/nvidia.mk

