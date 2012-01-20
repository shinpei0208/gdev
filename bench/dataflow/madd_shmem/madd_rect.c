#include <stdio.h>
#include <sys/time.h>
#include "madd.h"

#define GRAPH_ROWS 6
#define GRAPH_COLS 10
#define EXPECTED_SUM 7

#define INPUT_SIZE (((GRAPH_ROWS) + 1) * (GRAPH_COLS))

int perform_operation(struct device_info *device_info, int *input_matrix_keys,
 unsigned int begin_input, unsigned int end_input, unsigned int rows,
 unsigned int cols, int *r)
{
	int output_matrix_key;
	int output_matrix_shmid;
	int a_key;
	int b_key;
	unsigned long start_time, end_time;
	unsigned int i;

	CUresult res;

	int is_first_iter = 1;

	for (i = begin_input; i < end_input; i++) {
		if (is_first_iter) {
			a_key = input_matrix_keys[i];
			b_key = input_matrix_keys[i + 1];

			i++;

			is_first_iter = 0;
		}
		else {
			a_key = output_matrix_key;
			b_key = input_matrix_keys[i];
		}

		/* perform addition */
		if (madd_gpu(a_key, b_key, &output_matrix_key, rows, cols)) {
			*r = -1;
			return -1;
		}
	}

	return output_matrix_key;
}

int main(int argc, char **argv)
{
	unsigned int *input_matrices[INPUT_SIZE];
	unsigned int *output_matrices[GRAPH_COLS];
	int input_matrix_keys[INPUT_SIZE];
	int output_matrix_keys[GRAPH_COLS];

	unsigned int i, j, k;
	unsigned int begin_input, end_input;
	unsigned int rows, cols;

	int r = 0;
	int has_failed;

	struct timeval start_time, end_time;
	unsigned long start_us, end_us, total_time = 0;

	struct device_info device_info;

	int matrix_alloc_size;

        if (argc != 3) {
                fprintf(stderr, "Usage: %s rows cols\n", argv[0]);
                return -1;
        }

        rows = (unsigned int)atoi(argv[1]);
        cols = (unsigned int)atoi(argv[2]);

	matrix_alloc_size = rows * cols * sizeof(unsigned int);

        /* generate identity matrices */
        for (i = 0; i < INPUT_SIZE; i++) {
                if ((input_matrices[i] = (unsigned int *)
		 malloc(matrix_alloc_size)) == NULL) {
                        fprintf(stderr, "Out of memory error.\n");

                        for (j = 0; j < i; j++)
                                free(input_matrices[i]);

                        return -1;
                }

                for (j = 0; j < rows * cols; j++)
                        input_matrices[i][j] = 1;
        }

	/* begin GPU operations after recording start time */
	gettimeofday(&start_time, NULL);

	/* initialize GPU and begin GPU context */
        if (madd_gpu_init(&device_info)) {
                for (i = 0; i < INPUT_SIZE; i++)
                        free(input_matrices[i]);

                return -1;
        }


	begin_input = 0;
	end_input = GRAPH_ROWS + 1;

	/* Each column in the graph represents a parallel execution path. */
	for (i = 0; i < GRAPH_COLS; i++) {
		/*
		 * Allocate shared memory for relevant input matrices and copy
		 * input matrices from host memory to shared memory
		 */
		for (j = begin_input; j < end_input; j++) {
			input_matrix_keys[j] = key_counter++;

			if (shmem_device_copy(input_matrix_keys[i],
			 matrix_alloc_size, input_matrices[i], 1)) {
				for (k = 0; k < INPUT_SIZE; k++)
					free(input_matrices[j]);

				return -1;
			}
		}

		output_matrix_keys[i] = perform_operation(&device_info,
		 input_matrix_keys, begin_input, end_input, rows, cols, &r);


		if (r) {
			for (j = 0; j < INPUT_SIZE; j++)
				free(input_matrices[j]);

			return r;
		}

		/*
		 * Allocate memory for result matrix and copy it from shared
		 * memory to host memory
		 */
		if ((output_matrices[i] = (unsigned int *)
		 malloc(matrix_alloc_size)) == NULL) {
			fprintf(stderr, "Cannot allocate output matrices");

			for (j = 0; j < i; j++)
				free(output_matrices[j]);

			for (j = 0; j < INPUT_SIZE; j++)
				free(input_matrices[j]);

			return -1;
		}

		if (shmem_device_copy(output_matrix_keys[i], matrix_alloc_size,
		 output_matrices[i], 0)) {
			for (j = 0; j < i; j++)
				free(output_matrices[j]);

			for (j = 0; j < INPUT_SIZE; j++)
				free(input_matrices[j]);

			return -1;
		}

		/* Free shared memory for input and output matrices */
		for (j = begin_input; j < end_input; j++) {
			if (free_shmem(input_matrix_keys[j], matrix_alloc_size))
			{
				for (k = 0; k < i; k++)
					free(output_matrices[k]);

				for (k = 0; k < INPUT_SIZE; k++)
					free(input_matrices[k]);

				return -1;
			}
		}

		if (free_shmem(output_matrix_keys[i], matrix_alloc_size)) {
			for (k = 0; k < i; k++)
				free(output_matrices[k]);

			for (k = 0; k < INPUT_SIZE; k++)
				free(input_matrices[k]);

			return -1;
		}

		begin_input = end_input;
		end_input += GRAPH_ROWS + 1;
	}


	/* End GPU operations after closing context and record end time */
	if (madd_gpu_close(&device_info)) {
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
	for (i = 0; i < GRAPH_COLS && !has_failed; i++) {
		for (j = 0; j < rows * cols && !has_failed; j++) {
			if (output_matrices[i][j] != EXPECTED_SUM) {
				printf("output_matrix[%u][%u] was %u, but \\"
				 " expected %u\n", i, j, output_matrices[i][j],
				 EXPECTED_SUM);

				has_failed = 1;
			}
		}
	}

	printf("Test %s.\n", has_failed ? "failed" : "passed");
	printf("Running time of rectangle: %lu microseconds\n", total_time);

	/* free memory of input and output matrices */
	for (i = 0; i < INPUT_SIZE; i++)
		free(input_matrices[i]);

	for (i = 0; i < GRAPH_COLS; i++)
		free(output_matrices[i]);

	return 0;
}
