#include <stdio.h>
#include <sys/time.h>
#include "mmult.h"

#define GRAPH_ROWS 6
#define GRAPH_COLS 10

#define INPUT_SIZE (((GRAPH_ROWS) + 1) * (GRAPH_COLS))

CUdeviceptr perform_operation(struct device_info *device_info, CUdeviceptr 
 *input_dev_matrices, unsigned int begin_input, unsigned int end_input,
 unsigned int rows, unsigned int cols, int *r)
{
	CUdeviceptr output_dev_matrix;
	CUdeviceptr *a_dev;
	CUdeviceptr *b_dev;
	unsigned long start_time, end_time;
	unsigned int i;

	int is_first_iter = 1;

	/* allocate memory for output matrix */
	if (cuMemAlloc(&output_dev_matrix, rows * cols * sizeof(unsigned int))
	 != CUDA_SUCCESS) {
		printf("cuMemAlloc failed\n");

		*r = -1;

		return (CUdeviceptr)NULL;
	}

	for (i = begin_input; i < end_input; i++) {
		if (is_first_iter) {
			a_dev = &input_dev_matrices[i];
			b_dev = &input_dev_matrices[i + 1];

			i++;

			is_first_iter = 0;
		}
		else {
			a_dev = &output_dev_matrix;
			b_dev = &input_dev_matrices[i];
		}

		/* perform multiplication */
		if (mmult_gpu(device_info, a_dev, b_dev, &output_dev_matrix,
		 rows, cols)) {
			*r = -1;
			return (CUdeviceptr)NULL;
		}
	}

	return output_dev_matrix;
}

int main(int argc, char **argv)
{
	unsigned int *input_matrices[INPUT_SIZE];
	unsigned int *output_matrices[GRAPH_COLS];
	CUdeviceptr input_dev_matrices[INPUT_SIZE];
	CUdeviceptr output_dev_matrices[GRAPH_COLS];

	unsigned int i, j;
	unsigned int begin_input, end_input;
	unsigned int rows, cols;

	int r = 0;
	int has_failed;

	struct timeval start_time, end_time;
	unsigned long start_us, end_us, total_time = 0;

	struct device_info device_info;

	int matrix_alloc_size;
	CUresult res;

        if (argc != 3) {
                fprintf(stderr, "Usage: %s rows cols\n", argv[0]);
                return -1;
        }

        rows = (unsigned int)atoi(argv[1]);
        cols = (unsigned int)atoi(argv[2]);

	matrix_alloc_size = rows * cols * sizeof(unsigned int);

        /* generate matrices filled with 2s */
        for (i = 0; i < INPUT_SIZE; i++) {
                if ((input_matrices[i] = (unsigned int *)
		 malloc(matrix_alloc_size)) == NULL) {
                        fprintf(stderr, "Out of memory error.\n");

                        for (j = 0; j < i; j++)
                                free(input_matrices[j]);

                        return -1;
                }

                for (j = 0; j < rows * cols; j++)
                        input_matrices[i][j] = 2;
        }

	/* allocate space for output matrices */
	for (i = 0; i < GRAPH_COLS; i++) {
                if ((output_matrices[i] = (unsigned int *)
		 malloc(matrix_alloc_size)) == NULL) {
                        fprintf(stderr, "Out of memory error.\n");

                        for (j = 0; j < i; j++)
                                free(output_matrices[j]);

			for (j = 0; j < INPUT_SIZE; j++)
				free(input_matrices[j]);

                        return -1;
                }
	}

	/* start running GPU operations */
	gettimeofday(&start_time, NULL);

	if (mmult_gpu_init(&device_info)) {
		for (i = 0; i < INPUT_SIZE; i++)
			free(input_matrices[i]);

		for (i = 0; i < GRAPH_COLS; i++)
			free(output_matrices[i]);

		return -1;
	}

	/* copy input matrices from host memory to device memory */
	for (i = 0; i < INPUT_SIZE; i++) {
		if (cuMemAlloc(&input_dev_matrices[i], matrix_alloc_size) !=
		 CUDA_SUCCESS) {
			printf("cuMemAlloc failed\n");

                	for (i = 0; i < INPUT_SIZE; i++)
				free(input_matrices[i]);

			for (i = 0; i < GRAPH_COLS; i++)
				free(output_matrices[i]);

			return -1;
		}

		if ((res = cuMemcpyHtoD(input_dev_matrices[i],
		 input_matrices[i], matrix_alloc_size)) != CUDA_SUCCESS) {
			printf("cuMemcpyHtoD failed: res = %lu\n",
			 (unsigned long)res);

                	for (i = 0; i < INPUT_SIZE; i++)
				free(input_matrices[i]);

			for (i = 0; i < GRAPH_COLS; i++)
				free(output_matrices[i]);

			return -1;
		}
	}

	begin_input = 0;
	end_input = GRAPH_ROWS + 1;

	/* Each column in the graph represents a parallel execution path. */
	for (i = 0; i < GRAPH_COLS; i++) {
		output_dev_matrices[i] = perform_operation(&device_info,
		 input_dev_matrices, begin_input, end_input, rows, cols, &r);

		if (r) {
			for (j = 0; j < INPUT_SIZE; j++)
				free(input_matrices[j]);

			for (i = 0; i < GRAPH_COLS; i++)
				free(output_matrices[i]);

			return r;
		}

		begin_input = end_input;
		end_input += GRAPH_ROWS + 1;
	}

	/*
	 *  Copy output matrices from host memory to device memory and free
	 *  them.
	 */
	for (i = 0; i < GRAPH_COLS; i++) {
		if ((res = cuMemcpyDtoH(output_matrices[i],
		 output_dev_matrices[i], matrix_alloc_size)) != CUDA_SUCCESS) {
			printf("cuMemcpyDtoH failed: res = %lu\n",
			 (unsigned long)res);

                	for (i = 0; i < INPUT_SIZE; i++)
				free(input_matrices[i]);

			for (i = 0; i < GRAPH_COLS; i++)
				free(output_matrices[i]);

			return -1;
		}

		if ((res = cuMemFree(output_dev_matrices[i])) != CUDA_SUCCESS)
		{
			printf("cuMemFree failed: res = %lu\n",
			 (unsigned long)res);

                	for (i = 0; i < INPUT_SIZE; i++)
				free(input_matrices[i]);

			for (i = 0; i < GRAPH_COLS; i++)
				free(output_matrices[i]);

			return -1;
		}
	}

	/* Free input matrices in device memory */
	for (i = 0; i < INPUT_SIZE; i++) {
		if ((res = cuMemFree(input_dev_matrices[i])) != CUDA_SUCCESS) {
			printf("cuMemFree failed: res = %lu\n",
			 (unsigned long)res);

                	for (i = 0; i < INPUT_SIZE; i++)
				free(input_matrices[i]);

			for (i = 0; i < GRAPH_COLS; i++)
				free(output_matrices[i]);

			return -1;

		}
	}

	/* finish running GPU operations */
	if (mmult_gpu_close(&device_info)) {
		for (i = 0; i < INPUT_SIZE; i++)
			free(input_matrices[i]);

		for (i = 0; i < GRAPH_COLS; i++)
			free(output_matrices[i]);

		return -1;
	}

	gettimeofday(&end_time, NULL);

	start_us = (unsigned long)start_time.tv_sec * 1000000 + 
	 (unsigned long)start_time.tv_usec;
	end_us = (unsigned long)end_time.tv_sec * 1000000 +
	 (unsigned long)end_time.tv_usec;

	total_time = end_us - start_us;

	has_failed = 0;

	/* Test output matrices */
	/*
	for (i = 0; i < GRAPH_COLS && !has_failed; i++) {
		for (j = 0; j < rows * cols && !has_failed; j++) {
			if (output_matrices[i][j] != expected_product) {
				printf("output_matrix[%u][%u] was %u, but \\"
				 " expected %u\n", i, j, output_matrices[i][j],
				 expected_product);

				has_failed = 1;
			}
		}
	}

	printf("Test %s.\n", has_failed ? "failed" : "passed");
	*/
	printf("Running time of rectangle: %lu microseconds\n", total_time);

	for (i = 0; i < INPUT_SIZE; i++)
		free(input_matrices[i]);

	for (i = 0; i < GRAPH_COLS; i++)
		free(output_matrices[i]);

	return 0;
}
