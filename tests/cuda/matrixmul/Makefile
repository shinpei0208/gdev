# Makefile
TARGET	= user_test
CC	= gcc
NVCC	= nvcc -arch sm_20
LIBS	= -lcuda
CFLAGS	= -I /usr/local/cuda/include

all:
	#$(NVCC) main.c -o $(TARGET) $(CFLAGS) $(LIBS)
	gcc main.c -o $(TARGET) $(CFLAGS) $(LIBS)

clean:
	rm -f $(TARGET) ./*~
