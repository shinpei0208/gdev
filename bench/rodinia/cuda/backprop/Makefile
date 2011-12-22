#
# The following must be provided:
#

# A path to CUDA top directory
CUDA_TOP_DIR	:= ..
# An executable name
EXECUTABLE	:= backprop
# Cuda source files (compiled with cudacc)
CUFILES		:= backprop_cuda_kernel.cu
# C/C++ source files (compiled with gcc / c++)
CCFILES		:= backprop_cuda.c backprop.c facetrain.c imagenet.c

# init C flags (more options will be added)
CFLAGS		:= -lm

include ../util/gdev.mk
