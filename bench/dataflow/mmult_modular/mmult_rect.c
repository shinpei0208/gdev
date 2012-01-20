#include <stdio.h>
#include <sys/time.h>
#include "mmult.h"

#define GRAPH_ROWS 6
#define GRAPH_COLS 10
#define EXPECTED_SUM 7

#define INPUT_SIZE (((GRAPH_ROWS) + 1) * (GRAPH_COLS))

unsigned int *perform_operation(struct device_info *device_info, unsigned int
 **input_matrices, unsigned int begin_input, unsigned int end_input,
 unsigned int rows, unsigned int cols, int *r)
{
	unsigned int *output_matrix;
	unsigned int *a;
	unsigned int *b;
	unsigned long start_time, end_time;
	unsigned int i;

	int is_first_iter = 1;

	/* allocate memory for output matrix */
	if ((output_matrix = (unsigned int *)malloc(rows * cols *
	 sizeof(unsigned int))) == NULL) {
		fprintf(stderr, "Out of memory error.\n");

		*r = -1;

		return NULL;
	}

	for (i = begin_input; i < end_input; i++) {
		if (is_first_iter) {
			a = input_matrices[i];
			b = input_matrices[i + 1];

			i++;

			is_first_iter = 0;
		}
		else {
			a = output_matrix;
			b = input_matrices[i];
		}

		/* perform multiplication */
		if (mmult_gpu(device_info, a, b, output_matrix, rows, cols)) {
			free(output_matrix);

			*r = -1;
			return NULL;
		}
	}

	return output_matrix;
}

int main(int argc, char **argv)
{
	unsigned int *input_matrices[INPUT_SIZE];
	unsigned int *output_matrices[GRAPH_COLS];
	unsigned int i, j;
	unsigned int begin_input, end_input;
	unsigned int rows, cols;

	int r = 0;
	int has_failed;

	struct timeval start_time, end_time;
	unsigned long start_us, end_us, total_time = 0;

	struct device_info device_info;

        if (argc != 3) {
                fprintf(stderr, "Usage: %s rows cols\n", argv[0]);
                return -1;
        }

        rows = (unsigned int)atoi(argv[1]);
        cols = (unsigned int)atoi(argv[2]);

        /* generate identity matrices */
        for (i = 0; i < INPUT_SIZE; i++) {
                if ((input_matrices[i] = (unsigned int *)malloc(rows * cols *
                 sizeof(unsigned int))) == NULL) {
                        fprintf(stderr, "Out of memory error.\n");

                        for (j = 0; j < i; j++)
                                free(input_matrices[i]);

                        return -1;
                }

                for (j = 0; j < rows * cols; j++)
                        input_matrices[i][j] = 1;
        }

	begin_input = 0;
	end_input = GRAPH_ROWS + 1;

	/* Each column in the graph represents a parallel execution path. */
	for (i = 0; i < GRAPH_COLS; i++) {
		gettimeofday(&start_time, NULL);

		output_matrices[i] = perform_operation(&device_info,
		 input_matrices, begin_input, end_input, rows, cols, &r);

		gettimeofday(&end_time, NULL);

		if (r) {
			for (j = 0; j < INPUT_SIZE; j++)
				free(input_matrices[j]);

			for (j = 0; j < i; j++)
				free(output_matrices[j]);

			return r;
		}

		start_us = (unsigned long)start_time.tv_sec * 1000000 + 
		 (unsigned long)start_time.tv_usec;
		end_us = (unsigned long)end_time.tv_sec * 1000000 +
		 (unsigned long)end_time.tv_usec;

		total_time += end_us - start_us;

		begin_input = end_input;
		end_input += GRAPH_ROWS + 1;
	}

	has_failed = 0;

	/* Test output matrices */
	/*
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
	*/
	printf("Running time of rectangle: %lu microseconds\n", total_time);

	for (i = 0; i < INPUT_SIZE; i++)
		free(input_matrices[i]);

	for (i = 0; i < GRAPH_COLS; i++)
		free(output_matrices[i]);

	return 0;
}
