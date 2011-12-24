#
# The following must be provided:
#

# A path to CUDA top directory
CUDA_TOP_DIR	:= ..
# An executable name
EXECUTABLE	:= heartwall
# Cuda source files (compiled with nvcc)
CUFILES		:= heartwall_cuda_kernel.cu
# C/C++ source files (compiled with gcc / c++)
CCFILES		:= heartwall_cuda.c ./AVI/avilib.c ./AVI/avimod.c

# init C flags (more options will be added)
CFLAGS		:= -lm -I./AVI

include ../util/nvidia.mk

