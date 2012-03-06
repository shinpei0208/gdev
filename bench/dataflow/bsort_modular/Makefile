#######
#
# Author: Jason Aumiller
# Description: Makefile for bitonic sort micro benchmark
#
# nvcc gets the --ptxas-options=-v option so that the compiled function names
# in the .cu file are printed to stdout
#
#######

BIN	= bitonicSort
CC	= gcc
NVCC	= nvcc -arch sm_20 -cubin --ptxas-options=-v
LIBS	= -lgdev -lcuda -L /usr/local/gdev/lib64
CFLAGS	= -I /usr/local/gdev/include -g 


all: 
	$(NVCC) -o D_bitonic_sort_K.cubin D_bitonic_sort_K.cu
	gcc -o $(BIN) $(CFLAGS) $(LIBS) D_bitonic_sort.c bitonic_sort.c 

clean:
	rm -f $(BIN) *.cubin ./*~

again: clean all

