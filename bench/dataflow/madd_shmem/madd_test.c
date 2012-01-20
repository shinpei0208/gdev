#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "madd.h"

void fillMatrix(unsigned int *a, unsigned int rows, unsigned int cols, char c)
{
	unsigned int i;

	switch (c) {
	case 'i': 	/* identity matrix */
		for (i = 0; i < rows * cols; i++)
			a[i] = 1;

		break;
	case 'z':	/* zero matrix */
		for (i = 0; i < rows * cols; i++)
			a[i] = 0;

		break;
	default:	/* random matrix */
		for (i = 0; i < rows * cols; i++)
			a[i] = (unsigned int)(rand() % 50);

		break;
	}
}

void addMatrices(unsigned int *a, unsigned int *b, unsigned int *c, unsigned int
 rows, unsigned int cols)
{
	unsigned int i, j, idx;

	for (i = 0; i < rows; i++) {
		for (j = 0; j < cols; j++) {
			idx = get_element_index(i, j, cols);

			c[idx] = a[idx] + b[idx];
		}
	}
}

void freeMatrices(unsigned int *a, unsigned int *b, unsigned int *c, unsigned int
 *r)
{
	free(a);
	free(b);
	free(c);
	free(r);
}

int compareMatrices(unsigned int *a, unsigned int *b, unsigned int rows,
 unsigned int cols)
{
	int i;

	for (i = 0; i < rows * cols; i++) {
		if (a[i] != b[i])
			return 0;
	}

	return 1;
}

int main(int argc, char **argv)
{
	unsigned int *a;
	unsigned int *b;
	unsigned int *c;		/* contains result of a + b */
	unsigned int *r;		/* expected result of a + b */

	unsigned int rows; 
	unsigned int cols;

	char testChoice = argv[3][0];

	struct device_info device_info;

	if (argc != 4) {
		fprintf(stderr, "Usage: %s rows cols [i|z|r]\n", argv[0]);
		return -1;
	}

	rows = (unsigned int)atoi(argv[1]);
	cols = (unsigned int)atoi(argv[2]);

	srand((unsigned int)time(NULL));

	a = (unsigned int *)malloc(rows * cols * sizeof(unsigned int));
	b = (unsigned int *)malloc(rows * cols * sizeof(unsigned int));
	c = (unsigned int *)malloc(rows * cols * sizeof(unsigned int));
	r = (unsigned int *)malloc(rows * cols * sizeof(unsigned int));

	fillMatrix(a, rows, cols, testChoice);
	fillMatrix(b, rows, cols, testChoice);

	addMatrices(a, b, r, rows, cols);

	if (madd_gpu_init(&device_info)) {
		fprintf(stderr, "Error occurred while initializing CUDA.\n");

		freeMatrices(a, b, c, r);
		return -1;
	}

	if (madd_gpu(&device_info, a, b, c, rows, cols)) {
		fprintf(stderr, "Error occurred while using GPU.\n");

		freeMatrices(a, b, c, r);
		return -1;
	}

	if (madd_gpu_close(&device_info)) {
		fprintf(stderr, "Error occured while closing device.\n");

		freeMatrices(a, b, c, r);
		return -1;
	}

	if (compareMatrices(c, r, rows, cols))
		printf("Test passed.\n");
	else
		printf("Test failed.\n");

	freeMatrices(a, b, c, r);

	printf("Print me!\n");

	return 0;
}
