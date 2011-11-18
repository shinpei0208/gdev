# Makefile
TARGET	= user_test
CC	= gcc
NVCC	= nvcc -arch sm_20 -cubin
LIBS	= -lcuda -lgdev
CFLAGS	= -L /usr/local/gdev/lib64 -I /usr/local/gdev/include

all:
	$(NVCC) -o matrixmul_gpu.cubin matrixmul_gpu.cu
	gcc driver.c -o $(TARGET) $(CFLAGS) $(LIBS)

clean:
	rm -f $(TARGET) *.cubin ./*~
