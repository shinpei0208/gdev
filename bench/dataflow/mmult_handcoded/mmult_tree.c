#include <stdio.h>
#include <sys/time.h>
#include "mmult.h"

#define NUM_MATRICES 64		/* 64 unique matrices in the 6 x 32 tree */
#define EXPECTED_SUM 64 

CUdeviceptr *run_binary_tree(struct device_info *device_info, CUdeviceptr
 *input_dev_matrices, unsigned int num_input_matrices, unsigned int rows,
 unsigned int cols, int *r, unsigned int *num_product_matrices)
{
	CUdeviceptr *output_dev_matrices;
	CUdeviceptr *final_output_dev_matrices;
	CUdeviceptr a_dev;
	CUdeviceptr b_dev;
	unsigned int i, j, k, recursive_r;
	unsigned int num_output_matrices = num_input_matrices >> 1;
	int matrix_alloc_size = rows * cols * sizeof(unsigned int);
	CUresult res;

	/*
	 * If the number of input matrices is 1 and if the return code r is 0,
	 * then return the input matrix.
	 */
	if (num_input_matrices == 1 && !(*r)) {
		*num_product_matrices = 1;

		return input_dev_matrices;
	}

	/*
	 * If the return code is not 0, then return NULL.
	 */
	if (*r)
		return NULL;

	/* Allocate memory for array of output matrices. */
	if ((output_dev_matrices = (CUdeviceptr *)malloc(num_output_matrices *
	 sizeof(CUdeviceptr))) == NULL) {
		fprintf(stderr, "Out of memory error.\n");

		*r = -1;
		*num_product_matrices = 0;

		return NULL;
	}

	for (i = 0; i < num_output_matrices; i++) {
		if (cuMemAlloc(&output_dev_matrices[i], matrix_alloc_size) !=
		 CUDA_SUCCESS) {
			printf("cuMemAlloc failed\n");

			free(output_dev_matrices);

			*r = -1;
			*num_product_matrices = 0;

			return NULL;
		}
	}

	/*
	 * For each pair of input matrices, add them and store the product in the
	 * output matrix array.
	 */
	for (i = 0, j = 0; i < num_input_matrices; i += 2, j++) {
		/* printf("Adding %u and %u of %u matrices\n", i, i + 1,
		 num_input_matrices); */

		a_dev = input_dev_matrices[i];
		b_dev = input_dev_matrices[i + 1];

		/* perform multiplication */
		if (mmult_gpu(device_info, &a_dev, &b_dev,
		 &output_dev_matrices[j], rows, cols)) {
			free(output_dev_matrices);

			*r = -1;
			*num_product_matrices = 0;

			return NULL;
		}
	}

	/* number of product matrices == number of output matrices */
	*num_product_matrices = num_output_matrices;

	/* free input matrices */
	for (i = 0; i < num_input_matrices; i++) {
		if ((res = cuMemFree(input_dev_matrices[i])) != CUDA_SUCCESS) {
			printf("cuMemFree failed: res = %lu\n",
			 (unsigned long)res);

			free(output_dev_matrices);

			*r = -1;
			*num_product_matrices = 0;

			return NULL;
		}
	}

	free(input_dev_matrices);

	/* make recursive call on next level of binary tree */
	final_output_dev_matrices = run_binary_tree(device_info,
	 output_dev_matrices, num_output_matrices, rows, cols, r,
	 num_product_matrices);

	/* free intermediate output matrices */
	if (*num_product_matrices > 1) {
		for (i = 0; i < num_output_matrices; i++) {
			if ((res = cuMemFree(output_dev_matrices[i])) !=
			 CUDA_SUCCESS) {
				printf("cuMemFree failed: res = %lu\n",
				 (unsigned long)res);

				free(output_dev_matrices);

				*r = -1;
				*num_product_matrices = 0;

				return NULL;
			}
		}

		free(output_dev_matrices);
	}

	/* return the recursively-obtained product matrix list */
	return final_output_dev_matrices;
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
	CUdeviceptr *input_dev_matrices;
	CUdeviceptr *dev_product_matrices;
	unsigned int *product_matrix;

	struct device_info device_info;

	struct timeval start_time, end_time;
	unsigned long start_us, end_us;

	int matrix_alloc_size;
	unsigned long expected_product;

	CUresult res;

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
	matrix_alloc_size = rows * cols * sizeof(unsigned int);

	/* generate identity matrices */
	for (i = 0; i < num_input_matrices; i++) {
		if ((input_matrices[i] = (unsigned int *)
		 malloc(matrix_alloc_size)) == NULL) {
			fprintf(stderr, "Out of memory error.\n");

			for (j = 0; j < i; j++)
				free(input_matrices[i]);

			free(input_matrices);

			return -1;
		}

		for (j = 0; j < rows * cols; j++)
			input_matrices[i][j] = 1;
	}

	/* assume rows == cols in our experiments */
	/* expected_product = (rows << 1) + (cols << 1); */

	/*
	 * Begin GPU operations and set begin time
	 */
	gettimeofday(&start_time, NULL);

	/* load GPU */
	if (mmult_gpu_init(&device_info)) {
		for (i = 0; i < num_input_matrices; i++)
			free(input_matrices[i]);

		free(input_matrices);

		return -1;
	}

	/*
	 * Allocate device memory and copy input matrices from host memory to
	 * device memory.
	 */
	if ((input_dev_matrices = (CUdeviceptr *)malloc(num_input_matrices *
	 sizeof(CUdeviceptr))) == NULL) {
		fprintf(stderr,
		 "Cannot allocate memory for input_dev_matrices");

		for (i = 0; i < num_input_matrices; i++)
			free(input_matrices[i]);

		free(input_matrices);

		return -1;
	}

	for (i = 0; i < num_input_matrices; i++) {
		if (cuMemAlloc(&input_dev_matrices[i], matrix_alloc_size) !=
		 CUDA_SUCCESS) {
			printf("cuMemAlloc failed\n");

			for (i = 0; i < num_input_matrices; i++)
				free(input_matrices[i]);

			free(input_matrices);
			free(input_dev_matrices);

			return -1;
		}

		if ((res = cuMemcpyHtoD(input_dev_matrices[i],
		 input_matrices[i], matrix_alloc_size)) != CUDA_SUCCESS) {
			printf("cuMemcpyHtoD failed: res = %lu\n",
			 (unsigned long)res);

			for (i = 0; i < num_input_matrices; i++)
				free(input_matrices[i]);

			free(input_matrices);
			free(input_dev_matrices);

			return -1;
		}
	}

	dev_product_matrices = run_binary_tree(&device_info, input_dev_matrices,
	 num_input_matrices, rows, cols, &r, &num_product_matrices);

	/* ensure that there is only one product matrix */
	if (r || num_product_matrices != 1) {
		fprintf(stderr,
		 "Some error occurred while computing binary tree");
		fprintf(stderr, "r == %u, num_product_matrices == %u", r, 
		 num_product_matrices);

		free(dev_product_matrices);

		for (i = 0; i < num_input_matrices; i++)
			free(input_matrices[i]);

		free(input_matrices);

		return -1;
	}

	/* allocate local memory for product matrix */
	if ((product_matrix = (unsigned int *)malloc(matrix_alloc_size)) ==
	 NULL) {
		fprintf(stderr, "Cannot allocate memory for product_matrix\n");

		free(dev_product_matrices);

		for (i = 0; i < num_input_matrices; i++)
			free(input_matrices[i]);

		free(input_matrices);

		return -1;
	}

	/* copy product matrix from device memory to host memory */
	if ((res = cuMemcpyDtoH(product_matrix, dev_product_matrices[0],
	 matrix_alloc_size)) != CUDA_SUCCESS) {
		printf("cuMemcpyDtoH failed: res = %lu\n", (unsigned long)res);

		free(dev_product_matrices);

		for (i = 0; i < num_input_matrices; i++)
			free(input_matrices[i]);

		free(input_matrices);
		free(product_matrix);

		return -1;
	}

	/* free product matrix in device memory */
	if ((res = cuMemFree(dev_product_matrices[0])) != CUDA_SUCCESS) {
		printf("cuMemFree (dev_product_matrices[0]) failed: res= %lu\n",
		 (unsigned long)res);

		free(dev_product_matrices);

		for (i = 0; i < num_input_matrices; i++)
			free(input_matrices[i]);

		free(input_matrices);
		free(product_matrix);

		return -1;

	}

	free(dev_product_matrices);

	/* close GPU */
	if (mmult_gpu_close(&device_info)) {
		for (i = 0; i < num_input_matrices; i++)
			free(input_matrices[i]);

		free(input_matrices);

		free(product_matrix);

		return -1;
	}

	/*
	 * End GPU operations and set end time
	 */
	gettimeofday(&end_time, NULL);

	/* 
	 * Regarding the product matrix, ensure that all elements in it are 
	 * EXPECTED_SUM.
	 */
	has_failed = 0;

	/*
	for (i = 0; i < rows * cols && !has_failed; i++) {
		if (product_matrix[i] != expected_product) {
			printf("product_matrix[%u] was %u, but expected %u\n",
			 i, product_matrix[i], expected_product);	

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

	for (i = 0; i < num_input_matrices; i++)
		free(input_matrices[i]);

	free(input_matrices);

	free(product_matrix);

	return 0;
}
