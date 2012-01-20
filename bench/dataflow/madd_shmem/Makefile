# Makefile
TREE_TARGET	= madd_tree
RECT_TARGET	= madd_rect
TEST_TARGET	= madd_test
CC	= gcc
NVCC	= nvcc -arch sm_20 -cubin
LIBS	= -lgdev -lcuda -L /usr/local/gdev/lib64
CFLAGS	= -I /usr/local/gdev/include -g

all:
	$(NVCC) -o madd_gpu.cubin madd_gpu.cu
	gcc -o $(RECT_TARGET) $(CFLAGS) $(LIBS) madd_rect.c madd.c
#	gcc -o $(TEST_TARGET) $(CFLAGS) $(LIBS) madd_test.c madd.c
	gcc -o $(TREE_TARGET) $(CFLAGS) $(LIBS) madd_tree.c madd.c

clean:
#	rm -f $(TREE_TARGET) $(RECT_TARGET) $(TEST_TARGET) *.cubin ./*~
	rm -f $(TREE_TARGET) $(RECT_TARGET) *.cubin ./*~
