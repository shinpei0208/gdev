/*****************************************************************************
* Name        : D_bitonic_sort.c
* Author      : Jason Aumiller
* Version     :
* Copyright   :  
* Description : Implements bitonic sort using CUDA driver API
*				See D_bitonic_sort.h for configuration specific information.
/*****************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <sys/time.h>

#include "D_bitonic_sort.h"

/* Reads integers from stdin, one per line. */
int readIntegers(int *ray, uint array_size) {
	int i;
	char str_buf[32];
	uint ints_read = 0;

	/* Read from stdin */
	while (fgets(str_buf, sizeof(str_buf), stdin) != 0) {
		i=atoi(str_buf);
		ray[ints_read++] = i;

		/* Double if necessary */
		if ( ints_read >= array_size ) {
			array_size = array_size << 1;
			ray = (int*) realloc (ray, array_size * sizeof(int) );
		}
	}
	return ints_read;
}

/* Verifies that the elements in the array are sorted. */
int verifySort(int *ray, uint array_size) {
	uint errors=0;
	int i;
	for (i=0; i<array_size-1; ++i) {
		if (ray[i] > ray[i+1]) {
			++errors;
		}
	}
	return errors;
}

 
void print_usage(){
	printf("Usage: \n\tbitonicSort [-vO] -i | size | -p pow  \n\n");
	printf("Generates an array of 'size' integers (or 2^'pow') and sorts them. If the -i option is used \n");
	printf("then the array is read from stdin.\n");
	printf("\nOptions \n");
	printf("\t -i \t Read input from stdin until EOF.\n");
	printf("\t -O \t Print sorted list to stdout.\n");
	printf("\t -p \t Specify 'size' as a power of 2.\n");
	printf("\t -v \t Verbose.\n\n");
}

/* Generate array of random integers */
int randomInts(int *ray, uint array_size) {
	int i;
	srand(time(NULL));
	for (i=0; i<array_size; ++i) {
		ray[i] = rand() % 65536;
	}
}

int main( int argc, char** argv) {
	uint length;
	int *list;
	int status, verbose, c, i, j;
	int read_stdin;
	struct timeval start_time, end_time;
	unsigned long total_time;

	status = SUCCESS;
	verbose = 0;
	read_stdin = FALSE;
	length = 0;

	while ((c = getopt (argc, argv, "dip:vO")) != -1) {
		switch (c) {
		case 'd':
			verbose |= GROSS_DEBUG;
			break;
		case 'i':
			read_stdin = TRUE;
		case 'O':
			verbose |= OUTPUT;
			break;
		case 'p':
			length = 1 << atoi(optarg);
			break;
		case 'v':
			verbose |= DEBUG;
			break;
		case '?':
		default:
			print_usage();
			return FAILURE;
		}
	}

	/* Get the list to sort, either read from stdin or randomly generated */
	if ( read_stdin == TRUE ) {
		/* Read sequence of integers from stdin */
		list = (int*) malloc (INIT_INPUT_SIZE * sizeof(int) );
		length = readIntegers(list, INIT_INPUT_SIZE);
	} else if (optind >= argc) { /* No size was given */
		print_usage();
		return FAILURE;
	} else {
		/* Generate our own integers */
		if ( length==0 ) length = atoi(argv[optind]);
		list = (int*) malloc (length * sizeof(int) );
		randomInts(list, length);
	}

	/* Start timing */
	gettimeofday(&start_time, NULL);

	/***  Sort the list  ***/
	D_bitonic_sort(list, length);
	
	/* Stop timing */
	gettimeofday(&end_time, NULL);

	/* Calculate and display running time */
	total_time = ( (unsigned long)end_time.tv_sec * 1000000 + (unsigned long)end_time.tv_usec )
		   - ( (unsigned long)start_time.tv_sec * 1000000 +   (unsigned long)start_time.tv_usec   ) ;

	printf("GPU total running time: %lu microseconds.\n", total_time);

	/* Verify sort */
	if ( verbose & DEBUG ) printf("Verifying sort order.\n");

	j = verifySort(list,length);
	if ( j != 0 ){
		fprintf(stderr, "***\n***  List has %d elements out of order!\n***\n", j);
		status=FAILURE;
	} else if (verbose & DEBUG) printf("List is sorted.\n");

	/* Print sorted list */
	if ( verbose & OUTPUT ) {
		for (i=0; i < length; ++i) {
			printf("%d\n", list[i]);
		}
	}

	free(list);

	return status;
}
