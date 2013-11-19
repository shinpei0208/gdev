# Makefile
TARGET	= user_test
CC	= gcc
NVCC	= nvcc -arch sm_20 -cubin
LIBS	= -lucuda -lgdev
CFLAGS	= -L /usr/local/gdev/lib64 -I /usr/local/gdev/include

all:
	$(NVCC) -o loop_gpu.cubin loop_gpu.cu
	gcc -o $(TARGET) $(CFLAGS) main.c loop.c $(LIBS)

clean:
	rm -f $(TARGET) *.cubin ./*~
