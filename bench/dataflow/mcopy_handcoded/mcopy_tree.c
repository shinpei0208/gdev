#include <stdio.h>
#include <sys/time.h>
#include "mcopy.h"

#define NUM_MATRICES 64		/* 64 unique matrices in the 6 x 32 tree */
#define EXPECTED_SUM 64 

CUdeviceptr *run_binary_tree(struct device_info *device_info, CUdeviceptr
 *input_dev_matrices, unsigned int num_input_matrices, unsigned int rows,
 unsigned int cols, int *r, unsigned int *num_copy_matrices)
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
		*num_copy_matrices = 1;

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
		*num_copy_matrices = 0;

		return NULL;
	}

	for (i = 0; i < num_output_matrices; i++) {
		if (cuMemAlloc(&output_dev_matrices[i], matrix_alloc_size) !=
		 CUDA_SUCCESS) {
			printf("cuMemAlloc failed\n");

			free(output_dev_matrices);

			*r = -1;
			*num_copy_matrices = 0;

			return NULL;
		}
	}

	/*
	 * For each pair of input matrices, make a copy of the first one and
	 * store it in the output matrix array.
	 */
	for (i = 0, j = 0; i < num_input_matrices; i += 2, j++) {
		/* printf("Adding %u and %u of %u matrices\n", i, i + 1,
		 num_input_matrices); */

		a_dev = input_dev_matrices[i];
		b_dev = input_dev_matrices[i + 1];

		/* perform addition */
		if (mcopy_gpu(device_info, &a_dev, &b_dev,
		 &output_dev_matrices[j], rows, cols)) {
			free(output_dev_matrices);

			*r = -1;
			*num_copy_matrices = 0;

			return NULL;
		}
	}

	/* number of copy matrices == number of output matrices */
	*num_copy_matrices = num_output_matrices;

	/* free input matrices */
	for (i = 0; i < num_input_matrices; i++) {
		if ((res = cuMemFree(input_dev_matrices[i])) != CUDA_SUCCESS) {
			printf("cuMemFree failed: res = %lu\n",
			 (unsigned long)res);

			free(output_dev_matrices);

			*r = -1;
			*num_copy_matrices = 0;

			return NULL;
		}
	}

	free(input_dev_matrices);

	/* make recursive call on next level of binary tree */
	final_output_dev_matrices = run_binary_tree(device_info,
	 output_dev_matrices, num_output_matrices, rows, cols, r,
	 num_copy_matrices);

	/* free intermediate output matrices */
	if (*num_copy_matrices > 1) {
		for (i = 0; i < num_output_matrices; i++) {
			if ((res = cuMemFree(output_dev_matrices[i])) !=
			 CUDA_SUCCESS) {
				printf("cuMemFree failed: res = %lu\n",
				 (unsigned long)res);

				free(output_dev_matrices);

				*r = -1;
				*num_copy_matrices = 0;

				return NULL;
			}
		}

		free(output_dev_matrices);
	}

	/* return the recursively-obtained copy matrix list */
	return final_output_dev_matrices;
}

void free_matrices(unsigned int **input_matrices, unsigned int **copy_matrices,
 unsigned int num_input_matrices, unsigned int num_copy_matrices,
 int input_freed)
{
	unsigned int i;

	for (i = 0; i < num_copy_matrices; i++)
		free(copy_matrices[i]);

	free(copy_matrices);

	if (!input_freed) {
		for (i = 0; i < num_input_matrices; i++)
			free(input_matrices[i]);
	}

	free(input_matrices);
}

int main(int argc, char **argv)
{
	unsigned int i, j;
	unsigned int rows, cols, num_input_matrices, num_copy_matrices;

	int r = 0, has_failed;

	unsigned int **input_matrices;
	CUdeviceptr *input_dev_matrices;
	CUdeviceptr *dev_copy_matrices;
	unsigned int *copy_matrix;
	unsigned int *expected_matrix;

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

	/* generate identity matrices and expected matrix */
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

	if ((expected_matrix = (unsigned int *)malloc(matrix_alloc_size)) ==
	 NULL) {
		fprintf(stderr, "Out of memory error.\n");

		for (i = 0; i < num_input_matrices; i++)
			free(input_matrices[i]);

		free(input_matrices);
	}

	for (i = 0; i < rows * cols; i++)
		expected_matrix[i] = 1;

	/*
	 * Begin GPU operations and set begin time
	 */
	gettimeofday(&start_time, NULL);

	/* load GPU */
	if (mcopy_gpu_init(&device_info)) {
		for (i = 0; i < num_input_matrices; i++)
			free(input_matrices[i]);

		free(input_matrices);
		free(expected_matrix);

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
		free(expected_matrix);

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
			free(expected_matrix);

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
			free(expected_matrix);

			return -1;
		}
	}

	dev_copy_matrices = run_binary_tree(&device_info, input_dev_matrices,
	 num_input_matrices, rows, cols, &r, &num_copy_matrices);

	/* ensure that there is only one copy matrix */
	if (r || num_copy_matrices != 1) {
		fprintf(stderr,
		 "Some error occurred while computing binary tree");
		fprintf(stderr, "r == %u, num_copy_matrices == %u", r, 
		 num_copy_matrices);

		free(dev_copy_matrices);

		for (i = 0; i < num_input_matrices; i++)
			free(input_matrices[i]);

		free(input_matrices);
		free(expected_matrix);

		return -1;
	}

	/* allocate local memory for copy matrix */
	if ((copy_matrix = (unsigned int *)malloc(matrix_alloc_size)) == NULL) {
		fprintf(stderr, "Cannot allocate memory for copy_matrix\n");

		free(dev_copy_matrices);

		for (i = 0; i < num_input_matrices; i++)
			free(input_matrices[i]);

		free(input_matrices);
		free(expected_matrix);

		return -1;
	}

	/* copy copy matrix from device memory to host memory */
	if ((res = cuMemcpyDtoH(copy_matrix, dev_copy_matrices[0],
	 matrix_alloc_size)) != CUDA_SUCCESS) {
		printf("cuMemcpyDtoH failed: res = %lu\n", (unsigned long)res);

		free(dev_copy_matrices);

		for (i = 0; i < num_input_matrices; i++)
			free(input_matrices[i]);

		free(input_matrices);
		free(copy_matrix);
		free(expected_matrix);

		return -1;
	}

	/* free copy matrix in device memory */
	if ((res = cuMemFree(dev_copy_matrices[0])) != CUDA_SUCCESS) {
		printf("cuMemFree (dev_copy_matrices[0]) failed: res= %lu\n",
		 (unsigned long)res);

		free(dev_copy_matrices);

		for (i = 0; i < num_input_matrices; i++)
			free(input_matrices[i]);

		free(input_matrices);
		free(copy_matrix);
		free(expected_matrix);

		return -1;

	}

	free(dev_copy_matrices);

	/* close GPU */
	if (mcopy_gpu_close(&device_info)) {
		for (i = 0; i < num_input_matrices; i++)
			free(input_matrices[i]);

		free(input_matrices);

		free(copy_matrix);
		free(expected_matrix);

		return -1;
	}

	/*
	 * End GPU operations and set end time
	 */
	gettimeofday(&end_time, NULL);

	/* 
	 * Regarding the copy matrix, ensure that all elements in it are 
	 * equal to the expected matrix.
	 */
	has_failed = 0;

	for (i = 0; i < rows * cols && !has_failed; i++) {
		if (copy_matrix[i] != expected_matrix[i]) {
			printf("copy_matrix[%u] was %u, but expected %u\n",
			 i, copy_matrix[i], expected_matrix[i]);	

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

	free(copy_matrix);
	free(expected_matrix);

	return 0;
}
