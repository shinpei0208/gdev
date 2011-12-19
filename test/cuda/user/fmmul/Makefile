# Makefile
TARGET	= user_test
CC	= gcc
NVCC	= nvcc -arch sm_20 -cubin
LIBS	= -lcuda -lgdev
CFLAGS	= -L /usr/local/gdev/lib64 -I /usr/local/gdev/include

all:
	$(NVCC) -o fmmul_gpu.cubin fmmul_gpu.cu
	gcc -o $(TARGET) $(CFLAGS) $(LIBS) main.c fmmul.c

clean:
	rm -f $(TARGET) *.cubin ./*~
