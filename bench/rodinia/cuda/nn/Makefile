#
# The following must be provided:
#

# A path to CUDA top directory
CUDA_TOP_DIR	:= ..
# An executable name
EXECUTABLE	:= nn
# Cuda source files (compiled with nvcc)
CUFILES		:= nn_cuda_kernel.cu
# C/C++ source files (compiled with gcc / c++)
CCFILES		:= nn_cuda.c

# init C flags (more options will be added)
CFLAGS		:= -lm
# use g++ instead of gcc
CC		:= g++ 

include ../util/gdev.mk

hurricane_gen : hurricane_gen.c
	$(CC) -o $@ $< -lm

#data :
#       mkdir data
#       ./gen_dataset.sh

