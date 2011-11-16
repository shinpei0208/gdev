# Makefile
TARGET	= user_test
CC	= gcc
NVCC	= nvcc -arch sm_20
LIBS	= -lcuda -lgdev
CFLAGS	= -L /usr/local/gdev/lib64 -I /usr/local/gdev/include
SRC	= $(wildcard ./*.cu)

all:
	$(CC) main.c -o $(TARGET) $(CFLAGS) $(LIBS)

clean:
	rm -f $(TARGET) ./*~
