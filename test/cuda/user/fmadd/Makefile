# Makefile
TARGET	= user_test
CC	= gcc
NVCC	= nvcc -arch sm_20 -cubin
LIBS	= -lcuda -lgdev
CFLAGS	= -L /usr/local/gdev/lib64 -I /usr/local/gdev/include

all:
	$(NVCC) -o fmadd_gpu.cubin fmadd_gpu.cu
	gcc -o $(TARGET) $(CFLAGS) $(LIBS) main.c fmadd.c

clean:
	rm -f $(TARGET) *.cubin ./*~
