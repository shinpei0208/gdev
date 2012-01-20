#include <stdio.h>
#include <sys/time.h>
#include "mcopy.h"

#define NUM_MATRICES 64		/* 64 unique matrices in the 6 x 32 tree */
#define EXPECTED_SUM 64 

static int *run_binary_tree(struct device_info *device_info, int
 *input_matrix_keys, unsigned int num_input_matrices, unsigned int rows,
 unsigned int cols, int *r, unsigned int *num_copy_matrices)
{
	int *output_matrix_keys;
	int *final_output_matrix_keys;
	int a_key;
	int b_key;

	unsigned int i, j, k, recursive_r;
	unsigned int num_output_matrices = num_input_matrices >> 1;

	int current_shmid;
	CUresult res;

	int matrix_alloc_size = rows * cols * sizeof(unsigned int);

	/*
	 * If the number of input matrices is 1 and if the return code r is 0,
	 * then return the input matrix.
	 */
	if (num_input_matrices == 1 && !(*r)) {
		*num_copy_matrices = 1;

		return input_matrix_keys;
	}

	/*
	 * If the return code is not 0, then return NULL.
	 */
	if (*r)
		return NULL;

	/* Allocate memory for array of keys of output matrices. */
	if ((output_matrix_keys = (int *)malloc(num_output_matrices *
	 sizeof(int))) == NULL) {
		fprintf(stderr, "Out of memory error.\n");

		*r = -1;
		*num_copy_matrices = 0;

		return NULL;
	}

	/*
	 * For each pair of input matrices, add them and store the copy in the
	 * output matrix array.
	 */
	for (i = 0, j = 0; i < num_input_matrices; i += 2, j++) {
		/* printf("Adding %u and %u of %u matrices\n", i, i + 1,
		 num_input_matrices); */

		a_key = input_matrix_keys[i];
		b_key = input_matrix_keys[i + 1];

		/* perform addition */
		if (mcopy_gpu(a_key, b_key, &output_matrix_keys[j], rows,
		 cols)) {
			free(output_matrix_keys);

			*r = -1;
			*num_copy_matrices = 0;

			return NULL;
		}
	}

	/* number of copy matrices == number of output matrices */
	*num_copy_matrices = num_output_matrices;

	/* free input matrices in shared memory */
	for (i = 0; i < num_input_matrices; i++) {
		if (free_shmem(input_matrix_keys[i], matrix_alloc_size)) {
			free(output_matrix_keys);

			*r = -1;
			*num_copy_matrices = 0;

			return NULL;
		}
	}

	/* make recursive call on next level of binary tree */
	final_output_matrix_keys = run_binary_tree(device_info,
	 output_matrix_keys, num_output_matrices, rows, cols, r,
	 num_copy_matrices);

	/* free intermediate output matrices */
	if (*num_copy_matrices > 1) {
		for (i = 0; i < num_output_matrices; i++) {
			if (free_shmem(output_matrix_keys[i],
			 matrix_alloc_size)) {
				free(output_matrix_keys);
				free(final_output_matrix_keys);

				*r = -1;
				*num_copy_matrices = 0;

				return NULL;
			}
		}

		free(output_matrix_keys);
	}

	/* return the recursively-obtained copy matrix list */
	return final_output_matrix_keys;
}

void free_matrices(unsigned int **input_matrices, int *input_matrix_keys, int
 *copy_matrix_keys, unsigned int *copy_matrix, unsigned int num_input_matrices,
 int copy_matrix_allocated)
{
	unsigned int i;

	free(copy_matrix_keys);

	for (i = 0; i < num_input_matrices; i++)
		free(input_matrices[i]);

	free(input_matrices);
	free(input_matrix_keys);

	if (copy_matrix_allocated)
		free(copy_matrix);
}

int main(int argc, char **argv)
{
	unsigned int i, j;
	unsigned int rows, cols, num_input_matrices, num_copy_matrices;

	int r = 0, has_failed, matrix_alloc_size;

	unsigned int **input_matrices;
	unsigned int *copy_matrix;
	unsigned int *expected_matrix;

	int *input_matrix_keys;
	int *copy_matrix_keys;

	struct device_info device_info; /* this context is local */

	CUdeviceptr current_addr;
	int current_shmid;

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

	if ((input_matrix_keys = (int *)malloc(NUM_MATRICES * sizeof(int))) ==
	 NULL) {
		fprintf(stderr, "Out of memory error.\n");

		free(input_matrices);

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
			free(input_matrix_keys);

			return -1;
		}

		for (j = 0; j < rows * cols; j++)
			input_matrices[i][j] = 1;
	}

	if ((expected_matrix = (unsigned int *)malloc(matrix_alloc_size)) ==
	 NULL) {
		fprintf(stderr, "Out of memory error.\n");

		for (j = 0; j < i; j++)
			free(input_matrices[i]);

		free(input_matrices);
		free(input_matrix_keys);

		return -1;
	}

	for (i = 0; i < rows * cols; i++)
		expected_matrix[i] = 1;
	
	/* begin GPU operations after recording start time */
	gettimeofday(&start_time, NULL);

	/* initialize GPU and begin GPU context */
	if (mcopy_gpu_init(&device_info)) {
		for (j = 0; j < i; j++)
			free(input_matrices[i]);

		free(input_matrices);
		free(input_matrix_keys);

		free(expected_matrix);

		return -1;
	}

	/* copy input matrices from host memory to shared memory */
	for (i = 0; i < num_input_matrices; i++) {
		/* allocate shared memory */
		input_matrix_keys[i] = key_counter++;

		if (shmem_device_copy(input_matrix_keys[i], matrix_alloc_size,
		 input_matrices[i], 1)) {
			for (j = 0; j < num_input_matrices; j++)
				free(input_matrices[j]);

			free(input_matrices);
			free(input_matrix_keys);

			free(expected_matrix);

			return -1;
		}
	}

	/* execute the binary tree of additions */
	copy_matrix_keys = run_binary_tree(&device_info, input_matrix_keys,
	 num_input_matrices, rows, cols, &r, &num_copy_matrices);

	/* make sure there is only one copy matrix */
	if (r || num_copy_matrices != 1) {
		fprintf(stderr,
		 "Some error occurred while computing binary tree");
		fprintf(stderr, "r == %u, num_copy_matrices == %u", r, 
		 num_copy_matrices);

		free_matrices(input_matrices, input_matrix_keys,
		 copy_matrix_keys, copy_matrix, num_input_matrices, 0);
		free(expected_matrix);

		return -1;
	}

	/*
	 * Allocate memory for copy matrix and copy it from shared memory to
	 * host memory
	 */
	if ((copy_matrix = (unsigned int *)malloc(matrix_alloc_size)) == NULL) {
		free_matrices(input_matrices, input_matrix_keys,
		 copy_matrix_keys, copy_matrix, num_input_matrices, 0);
		free(expected_matrix);

		return -1;
	}

	if (shmem_device_copy(copy_matrix_keys[0], matrix_alloc_size, copy_matrix,
	 0)) {
		free_matrices(input_matrices, input_matrix_keys,
		 copy_matrix_keys, copy_matrix, num_input_matrices, 1);
		free(expected_matrix);

		return -1;
	}

	/* end GPU operations after closing context and recording end time */
	if (mcopy_gpu_close(&device_info)) {
		free_matrices(input_matrices, input_matrix_keys,
		 copy_matrix_keys, copy_matrix, num_input_matrices, 1);
		free(expected_matrix);

		return -1;
	}

	gettimeofday(&end_time, NULL);

	/* 
	 * Regarding the copy matrix, ensure that all elements in it are 
	 * expected_matrix.
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

	free_matrices(input_matrices, input_matrix_keys, copy_matrix_keys,
	 copy_matrix, num_input_matrices, 1);
	free(expected_matrix);

	return 0;
}
