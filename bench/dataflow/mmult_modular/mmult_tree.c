#include <stdio.h>
#include <sys/time.h>
#include "mmult.h"

#define NUM_MATRICES 64		/* 64 unique matrices in the 6 x 32 tree */
#define EXPECTED_SUM 64 

unsigned int **run_binary_tree(struct device_info *device_info, unsigned int
 **input_matrices, unsigned int num_input_matrices, unsigned int rows,
 unsigned int cols, int *r, unsigned int *num_product_matrices)
{
	unsigned int **output_matrices;
	unsigned int **final_output_matrices;
	unsigned int *a;
	unsigned int *b;
	unsigned int i, j, k, recursive_r;
	unsigned int num_output_matrices = num_input_matrices >> 1;

	/*
	 * If the number of input matrices is 1 and if the return code r is 0,
	 * then return the input matrix.
	 */
	if (num_input_matrices == 1 && !(*r)) {
		*num_product_matrices = 1;

		return input_matrices;
	}

	/*
	 * If the return code is not 0, then return NULL.
	 */
	if (*r)
		return NULL;

	/* Allocate memory for array of output matrices. */
	if ((output_matrices = (unsigned int **)malloc(num_output_matrices *
	 sizeof(unsigned int *))) == NULL) {
		fprintf(stderr, "Out of memory error.\n");

		*r = -1;
		*num_product_matrices = 0;

		return NULL;
	}

	/*
	 * For each pair of input matrices, add them and store the product in the
	 * output matrix array.
	 */
	for (i = 0, j = 0; i < num_input_matrices; i += 2, j++) {
		/* printf("Adding %u and %u of %u matrices\n", i, i + 1,
		 num_input_matrices); */

		a = input_matrices[i];
		b = input_matrices[i + 1];

		/* allocate memory for output */
		if ((output_matrices[j] = (unsigned int *)malloc(rows * cols *
		 sizeof(unsigned int))) == NULL) {
			fprintf(stderr, "Out of memory error.\n");

			free(output_matrices);

			*r = -1;
			*num_product_matrices = 0;

			return NULL;
		}

		/* perform addition */
		if (mmult_gpu(device_info, a, b, output_matrices[j], rows,
		 cols)) {
			for (k = 0; k < j; k++)
				free(output_matrices[k]);

			free(output_matrices);

			*r = -1;
			*num_product_matrices = 0;

			return NULL;
		}
	}

	/* number of product matrices == number of output matrices */
	*num_product_matrices = num_output_matrices;

	/* free input matrices */
	for (i = 0; i < num_input_matrices; i++)
		free(input_matrices[i]);

	/* make recursive call on next level of binary tree */
	final_output_matrices = run_binary_tree(device_info, output_matrices,
	 num_output_matrices, rows, cols, r, num_product_matrices);

	/* free intermediate output matrices */
	if (*num_product_matrices > 1) {
		for (i = 0; i < num_output_matrices; i++)
			free(output_matrices[i]);

		free(output_matrices);
	}

	/* return the recursively-obtained product matrix list */
	return final_output_matrices;
}

void free_matrices(unsigned int **input_matrices, unsigned int **product_matrices,
 unsigned int num_input_matrices, unsigned int num_product_matrices,
 int input_freed)
{
	unsigned int i;

	for (i = 0; i < num_product_matrices; i++)
		free(product_matrices[i]);

	free(product_matrices);

	if (!input_freed) {
		for (i = 0; i < num_input_matrices; i++)
			free(input_matrices[i]);
	}

	free(input_matrices);
}

int main(int argc, char **argv)
{
	unsigned int i, j;
	unsigned int rows, cols, num_input_matrices, num_product_matrices;

	int r = 0, has_failed;

	unsigned int **input_matrices;
	unsigned int **product_matrix;

	struct device_info device_info;

	struct timeval start_time, end_time;
	unsigned long start_us, end_us;

	if (argc != 3) {
		fprintf(stderr, "Usage: %s rows cols\n", argv[0]);
		return -1;
	}

	rows = (unsigned int)atoi(argv[1]);
	cols = (unsigned int)atoi(argv[2]);

	if ((input_matrices = (unsigned int **)malloc(NUM_MATRICES *
	 sizeof(unsigned int *))) == NULL) {
		fprintf(stderr, "Out of memory error.\n");
		return -1;
	}

	num_input_matrices = NUM_MATRICES;

	/* generate identity matrices */
	for (i = 0; i < num_input_matrices; i++) {
		if ((input_matrices[i] = (unsigned int *)malloc(rows * cols *
		 sizeof(unsigned int))) == NULL) {
			fprintf(stderr, "Out of memory error.\n");

			for (j = 0; j < i; j++)
				free(input_matrices[i]);

			free(input_matrices);

			return -1;
		}

		for (j = 0; j < rows * cols; j++)
			input_matrices[i][j] = 1;
	}

	/* execute the binary tree of additions */
	gettimeofday(&start_time, NULL);

	product_matrix = run_binary_tree(&device_info, input_matrices,
	 num_input_matrices, rows, cols, &r, &num_product_matrices);

	gettimeofday(&end_time, NULL);

	if (r || num_product_matrices != 1) {
		fprintf(stderr,
		 "Some error occurred while computing binary tree");
		fprintf(stderr, "r == %u, num_product_matrices == %u", r, 
		 num_product_matrices);

		free_matrices(input_matrices, product_matrix, num_input_matrices,
		 num_product_matrices, 0);

		return -1;
	}

	/* 
	 * Regarding the product matrix, ensure that all elements in it are 
	 * EXPECTED_SUM.
	 */
	/*
	has_failed = 0;

	for (i = 0; i < rows * cols && !has_failed; i++) {
		if (product_matrix[0][i] != EXPECTED_SUM) {
			printf("product_matrix[%u] was %u, but expected %u\n",
			 i, product_matrix[0][i], EXPECTED_SUM);	

			has_failed = 1;
		}
	}

	printf("Test %s.\n", has_failed ? "failed" : "passed");
	*/

	start_us = (unsigned long)start_time.tv_sec * 1000000 +
	 (unsigned long)start_time.tv_usec;
	end_us = (unsigned long)end_time.tv_sec * 1000000 +
	 (unsigned long)end_time.tv_usec;

	printf("Running time of binary search tree: %lu microseconds\n",
	 end_us - start_us);

	free_matrices(input_matrices, product_matrix, num_input_matrices, 1, 1);

	return 0;
}
