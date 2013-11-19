# Makefile
TARGET	= user_test
CC	= gcc
NVCC	= nvcc -arch sm_20 -cubin
LIBS	= -lucuda -lgdev
CFLAGS	= -L /usr/local/gdev/lib64 -I /usr/local/gdev/include

all:
	$(NVCC) -o mmul_gpu.cubin mmul_gpu.cu
	gcc -o $(TARGET) $(CFLAGS) main.c mmul_vmmap_hybrid.c $(LIBS)

clean:
	rm -f $(TARGET) *.cubin ./*~
