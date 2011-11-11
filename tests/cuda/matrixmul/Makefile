# Makefile
TARGET	= user_test
CC	= gcc
NVCC	= nvcc -arch sm_20
LIBS	= -lcuda
CFLAGS	= -I /usr/local/cuda/include

all:
	#$(NVCC) runtime.c -o $(TARGET).rt $(CFLAGS) $(LIBS)
	gcc driver.c -o $(TARGET).drv $(CFLAGS) $(LIBS)

clean:
	rm -f $(TARGET).drv $(TARGET).rt ./*~
