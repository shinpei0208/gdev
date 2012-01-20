# Makefile
TREE_TARGET	= mcopy_tree
RECT_TARGET	= mcopy_rect
CC	= gcc
NVCC	= nvcc -arch sm_20 -cubin
LIBS	= -lgdev -lcuda -L /usr/local/gdev/lib64
CFLAGS	= -I /usr/local/gdev/include -g

all:
	gcc -o $(RECT_TARGET) $(CFLAGS) $(LIBS) mcopy_rect.c mcopy.c
	gcc -o $(TREE_TARGET) $(CFLAGS) $(LIBS) mcopy_tree.c mcopy.c

clean:
	rm -f $(TREE_TARGET) $(RECT_TARGET) *.cubin ./*~
