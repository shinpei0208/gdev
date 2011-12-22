#
# The following must be provided:
#

# A path to CUDA top directory
CUDA_TOP_DIR	:= ../..
# An executable name
EXECUTABLE	:= lud
# Cuda source files (compiled with nvcc)
CUFILES		:= lud_cuda_kernel.cu
# C/C++ source files (compiled with gcc / c++)
CCFILES		:= lud_cuda.c ../common/common.c

# init C flags (more options will be added)
CFLAGS		:= -I../common -lm

include $(CUDA_TOP_DIR)/util/nvidia.mk
