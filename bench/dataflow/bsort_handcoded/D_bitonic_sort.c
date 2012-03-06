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
#include <cuda.h>
#include <sys/time.h>

#include "D_bitonic_sort.h"

/* Reads integers from stdin, one per line. */
int readIntegers(int *ray, uint array_size) {
	int i;
	char str_buf[32];
	uint ints_read = 0;

	// Read from stdin
	while (fgets(str_buf, sizeof(str_buf), stdin) != 0) {
		i=atoi(str_buf);
		ray[ints_read++] = i;

		// Double if necessary
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

int main( int argc, char** argv)
{
	uint num_threads;
	uint num_blocks, block_size;
	uint length;
	uint nBytes;
	int *list;
	int status, verbose, c, i, j, logBlocks;
	int read_stdin;
	struct timeval start_time, end_time;
	unsigned long total_time;
	CUdevice hDevice;
	CUcontext hContext;
	CUmodule hModule;
	CUfunction bitonicBlockFn;
	CUfunction mergeBlocksFn;
	CUdeviceptr pDeviceArrayA;
	CUdeviceptr pDeviceArrayB;

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

	if ( read_stdin == TRUE ) {
		/* Read sequence of integers from stdin */
		list = (int*) malloc (INIT_INPUT_SIZE * sizeof(int) );
		length = readIntegers(list, INIT_INPUT_SIZE);
	} else if ( length > 0 ) {
		list = (int*) malloc (length * sizeof(int) );
		randomInts(list, length);
	} else if (optind >= argc) { /* No size was given */
		print_usage();
		return FAILURE;
	} else {
		/* Generate our own integers */
		length = atoi(argv[optind]);
		list = (int*) malloc (length * sizeof(int) );
		randomInts(list, length);
	}

	/*
	* Phase 1:
	* 	There will be one thread for each element to be sorted. Each
	*	block will perform bitonic sort on MAX_THREADS_PER_BLOCK elements.
	*/

	/* Initialize sizes */
	num_threads = _min(length, MAX_THREADS_PER_BLOCK );
	num_blocks = (length-1) / MAX_THREADS_PER_BLOCK + 1;
	nBytes = length * sizeof(int);

	if (verbose & DEBUG) printf("Initializing GPU.\n");
	
	/* Start timing */
	gettimeofday(&start_time, NULL);

	/* Initialize GPU */
	cutilDrvSafeCall( cuInit(0) 					);
	cutilDrvSafeCall( cuDeviceGet(&hDevice, 0)			); 
	cutilDrvSafeCall( cuCtxCreate(&hContext, 0, hDevice) 		);
	cutilDrvSafeCall( cuModuleLoad(&hModule, MODULE_FILE) 		);
	cutilDrvSafeCall( cuModuleGetFunction(&bitonicBlockFn, hModule, BITONIC_BLOCK_FN) );

	/* Allocate memory on the device */
	cutilDrvSafeCall( cuMemAlloc(&pDeviceArrayA, nBytes)		);
	cutilDrvSafeCall( cuMemAlloc(&pDeviceArrayB, nBytes)		);
	cutilDrvSafeCall( cuMemcpyHtoD(pDeviceArrayA, list, nBytes) 	);
	cutilDrvSafeCall( cuFuncSetBlockShape(bitonicBlockFn, num_threads, 1, 1));
	cutilDrvSafeCall( cuParamSeti(bitonicBlockFn, 0, pDeviceArrayA)	);
	cutilDrvSafeCall( cuParamSetSize(bitonicBlockFn, 4)		);
	
	/* Execute the kernel on the GPU */
	if ( verbose & DEBUG ) printf("Launching bitonic sort kernel with %d blocks and %d threads per block.\n", num_blocks, num_threads);
	cutilDrvSafeCall( cuLaunchGrid(bitonicBlockFn, num_blocks, 1)		);

	/*
	* Phase 2:
	* 	At this point each block is a sorted list. Now it's time to merge them.	
	*/

	/* TODO This should go away after development */
	if ( verbose & GROSS_DEBUG ) {
		cuMemcpyDtoH(list, pDeviceArrayA, nBytes);
		for (i=0; i<num_blocks; ++i) {
			printf("### Block %d:\n", i);
			for (j=0; j<num_threads; ++j) {
				printf("%d\n", list[i*num_threads + j]);
			}
		}
	}
	
	i=0;

	/* Do we need to merge blocks? */
	if ( num_blocks > 1 ) {

		/* There will be Log_2(num_blocks) merge steps. */
		logBlocks = 0;
		for (i=1; i<num_blocks; i *= 2)	++logBlocks;

		if ( verbose & DEBUG ) printf("There will be %d merge steps.\n", logBlocks);	

		block_size = num_threads; 	/* How big the blocks were in the last grid launch. */
		num_threads = num_blocks >> 1;  /* Start with blocks/2 threads */
		num_blocks = (num_threads-1) / MAX_THREADS_PER_BLOCK  +  1;

		cutilDrvSafeCall( cuModuleGetFunction(&mergeBlocksFn, hModule, MERGE_BLOCKS_FN) );
		cuParamSeti(mergeBlocksFn, 4, block_size);
		cuParamSetSize(mergeBlocksFn, 16);

		for (i=0; i < logBlocks; ++i) {
			cuFuncSetBlockShape(mergeBlocksFn, num_threads, 1, 1);
			cuParamSeti(mergeBlocksFn, 0, i); /* set merge level */

			/* Merging uses a source array and destination array, the gpu has 2 arrays allocated
			 * so we swap which is the source and which is the destination for each iteration. */
			if ( i%2 == 0 ) {
				cuParamSeti(mergeBlocksFn, 8, pDeviceArrayA);
				cuParamSeti(mergeBlocksFn, 12, pDeviceArrayB);
			} else {
				cuParamSeti(mergeBlocksFn, 8, pDeviceArrayB);
				cuParamSeti(mergeBlocksFn, 12, pDeviceArrayA);
			}

			if ( verbose & DEBUG ) {
				printf("Launching block merge kernel with %d blocks and %d threads per block\n", 
									num_blocks, num_threads/num_blocks);
			}	
			cutilDrvSafeCall( cuLaunchGrid(mergeBlocksFn, num_blocks, 1) );

			num_threads = num_threads >> 1;
			num_blocks = (num_threads-1) / MAX_THREADS_PER_BLOCK  +  1;
		}
	}

	/* Determine if answer is in pDeviceArrayA or pDeviceArrayB and copy to host */
	if (i%2 == 0) 	cutilDrvSafeCall( cuMemcpyDtoH(list, pDeviceArrayA, nBytes) );
	else			cutilDrvSafeCall( cuMemcpyDtoH(list, pDeviceArrayB, nBytes) );

	cutilDrvSafeCall( cuMemFree(pDeviceArrayA)	);
	cutilDrvSafeCall( cuMemFree(pDeviceArrayB)	);
  	cutilDrvSafeCall( cuModuleUnload(hModule)	);
  	cutilDrvSafeCall( cuCtxDestroy(hContext) 	);
	
	/* Stop timing */
	gettimeofday(&end_time, NULL);

	total_time = ( (unsigned long)end_time.tv_sec * 1000000 + (unsigned long)end_time.tv_usec )
		   - ( (unsigned long)start_time.tv_sec * 1000000 +   (unsigned long)start_time.tv_usec   ) ;

	printf("GPU total running time: %lu microseconds.\n", total_time);

	if ( verbose & DEBUG ) printf("Verifying sort order.\n");

	j = verifySort(list,length);
	if ( j != 0 ){
		fprintf(stderr, "***\n***  List has %d elements out of order!\n***\n", j);
		status=FAILURE;
	} else if (verbose & DEBUG) printf("List is sorted.\n");

	if ( verbose & OUTPUT ) {
		for (i=0; i < length; ++i) {
			printf("%d\n", list[i]);
		}
	}

	free(list);

	return status;
}
