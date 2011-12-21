#
# The following must be provided:
#

# A path to CUDA top directory
CUDA_TOP_DIR	:= ../..
# An executable name
EXECUTABLE	:= lud
# Cuda source files (compiled with cudacc)
CUFILES		:= lud_kernel.cu
# C/C++ source files (compiled with gcc / c++)
CCFILES		:= lud.c ../common/common.c
# C flags
CFLAGS		:= -I../common

include $(CUDA_TOP_DIR)/util/gdev.mk
