#include <stdio.h>
#include <sys/time.h>
#include "madd.h"

#define NUM_MATRICES 64		/* 64 unique matrices in the 6 x 32 tree */
#define EXPECTED_SUM 64 

CUdeviceptr *run_binary_tree(struct device_info *device_info, CUdeviceptr
 *input_dev_matrices, unsigned int num_input_matrices, unsigned int rows,
 unsigned int cols, int *r, unsigned int *num_sum_matrices)
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
		*num_sum_matrices = 1;

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
		*num_sum_matrices = 0;

		return NULL;
	}

	for (i = 0; i < num_output_matrices; i++) {
		if (cuMemAlloc(&output_dev_matrices[i], matrix_alloc_size) !=
		 CUDA_SUCCESS) {
			printf("cuMemAlloc failed\n");

			free(output_dev_matrices);

			*r = -1;
			*num_sum_matrices = 0;

			return NULL;
		}
	}

	/*
	 * For each pair of input matrices, add them and store the sum in the
	 * output matrix array.
	 */
	for (i = 0, j = 0; i < num_input_matrices; i += 2, j++) {
		/* printf("Adding %u and %u of %u matrices\n", i, i + 1,
		 num_input_matrices); */

		a_dev = input_dev_matrices[i];
		b_dev = input_dev_matrices[i + 1];

		/* perform addition */
		if (madd_gpu(device_info, &a_dev, &b_dev,
		 &output_dev_matrices[j], rows, cols)) {
			free(output_dev_matrices);

			*r = -1;
			*num_sum_matrices = 0;

			return NULL;
		}
	}

	/* number of sum matrices == number of output matrices */
	*num_sum_matrices = num_output_matrices;

	/* free input matrices */
	for (i = 0; i < num_input_matrices; i++) {
		if ((res = cuMemFree(input_dev_matrices[i])) != CUDA_SUCCESS) {
			printf("cuMemFree failed: res = %lu\n",
			 (unsigned long)res);

			free(output_dev_matrices);

			*r = -1;
			*num_sum_matrices = 0;

			return NULL;
		}
	}

	free(input_dev_matrices);

	/* make recursive call on next level of binary tree */
	final_output_dev_matrices = run_binary_tree(device_info,
	 output_dev_matrices, num_output_matrices, rows, cols, r,
	 num_sum_matrices);

	/* free intermediate output matrices */
	if (*num_sum_matrices > 1) {
		for (i = 0; i < num_output_matrices; i++) {
			if ((res = cuMemFree(output_dev_matrices[i])) !=
			 CUDA_SUCCESS) {
				printf("cuMemFree failed: res = %lu\n",
				 (unsigned long)res);

				free(output_dev_matrices);

				*r = -1;
				*num_sum_matrices = 0;

				return NULL;
			}
		}

		free(output_dev_matrices);
	}

	/* return the recursively-obtained sum matrix list */
	return final_output_dev_matrices;
}

void free_matrices(unsigned int **input_matrices, unsigned int **sum_matrices,
 unsigned int num_input_matrices, unsigned int num_sum_matrices,
 int input_freed)
{
	unsigned int i;

	for (i = 0; i < num_sum_matrices; i++)
		free(sum_matrices[i]);

	free(sum_matrices);

	if (!input_freed) {
		for (i = 0; i < num_input_matrices; i++)
			free(input_matrices[i]);
	}

	free(input_matrices);
}

int main(int argc, char **argv)
{
	unsigned int i, j;
	unsigned int rows, cols, num_input_matrices, num_sum_matrices;

	int r = 0, has_failed;

	unsigned int **input_matrices;
	CUdeviceptr *input_dev_matrices;
	CUdeviceptr *dev_sum_matrices;
	unsigned int *sum_matrix;

	struct device_info device_info;

	struct timeval start_time, end_time;
	unsigned long start_us, end_us;

	int matrix_alloc_size;

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

	/*
	 * Begin GPU operations and set begin time
	 */
	gettimeofday(&start_time, NULL);

	/* load GPU */
	if (madd_gpu_init(&device_info)) {
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

	dev_sum_matrices = run_binary_tree(&device_info, input_dev_matrices,
	 num_input_matrices, rows, cols, &r, &num_sum_matrices);

	/* ensure that there is only one sum matrix */
	if (r || num_sum_matrices != 1) {
		fprintf(stderr,
		 "Some error occurred while computing binary tree");
		fprintf(stderr, "r == %u, num_sum_matrices == %u", r, 
		 num_sum_matrices);

		free(dev_sum_matrices);

		for (i = 0; i < num_input_matrices; i++)
			free(input_matrices[i]);

		free(input_matrices);

		return -1;
	}

	/* allocate local memory for sum matrix */
	if ((sum_matrix = (unsigned int *)malloc(matrix_alloc_size)) == NULL) {
		fprintf(stderr, "Cannot allocate memory for sum_matrix\n");

		free(dev_sum_matrices);

		for (i = 0; i < num_input_matrices; i++)
			free(input_matrices[i]);

		free(input_matrices);

		return -1;
	}

	/* copy sum matrix from device memory to host memory */
	if ((res = cuMemcpyDtoH(sum_matrix, dev_sum_matrices[0],
	 matrix_alloc_size)) != CUDA_SUCCESS) {
		printf("cuMemcpyDtoH failed: res = %lu\n", (unsigned long)res);

		free(dev_sum_matrices);

		for (i = 0; i < num_input_matrices; i++)
			free(input_matrices[i]);

		free(input_matrices);
		free(sum_matrix);

		return -1;
	}

	/* free sum matrix in device memory */
	if ((res = cuMemFree(dev_sum_matrices[0])) != CUDA_SUCCESS) {
		printf("cuMemFree (dev_sum_matrices[0]) failed: res= %lu\n",
		 (unsigned long)res);

		free(dev_sum_matrices);

		for (i = 0; i < num_input_matrices; i++)
			free(input_matrices[i]);

		free(input_matrices);
		free(sum_matrix);

		return -1;

	}

	free(dev_sum_matrices);

	/* close GPU */
	if (madd_gpu_close(&device_info)) {
		for (i = 0; i < num_input_matrices; i++)
			free(input_matrices[i]);

		free(input_matrices);

		free(sum_matrix);

		return -1;
	}

	/*
	 * End GPU operations and set end time
	 */
	gettimeofday(&end_time, NULL);

	/* 
	 * Regarding the sum matrix, ensure that all elements in it are 
	 * EXPECTED_SUM.
	 */
	has_failed = 0;

	for (i = 0; i < rows * cols && !has_failed; i++) {
		if (sum_matrix[i] != EXPECTED_SUM) {
			printf("sum_matrix[%u] was %u, but expected %u\n",
			 i, sum_matrix[i], EXPECTED_SUM);	

			has_failed = 1;
		}
	}

	printf("Test %s.\n", has_failed ? "failed" : "passed");

	start_us = (unsigned long)start_time.tv_sec * 1000000 +
	 (unsigned long)start_time.tv_usec;
	end_us = (unsigned long)end_time.tv_sec * 1000000 +
	 (unsigned long)end_time.tv_usec;

	printf("Running time of binary search tree: %lu microseconds\n",
	 end_us - start_us);

	for (i = 0; i < num_input_matrices; i++)
		free(input_matrices[i]);

	free(input_matrices);

	free(sum_matrix);

	return 0;
}
