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
#include "myutil.h"

#define VERB DEBUG

void gpu_init(struct device_info *dev) {
	cutilDrvSafeCall( cuInit(0) 								);
	cutilDrvSafeCall( cuDeviceGet(&dev->dev, 0)					); 
	cutilDrvSafeCall( cuCtxCreate(&dev->context, 0, dev->dev)	);
	cutilDrvSafeCall( cuModuleLoad(&dev->module, MODULE_FILE) 	);
	cutilDrvSafeCall( cuModuleGetFunction(&dev->kernel1, dev->module, BITONIC_BLOCK_FN) );
	cutilDrvSafeCall( cuModuleGetFunction(&dev->kernel2, dev->module, MERGE_BLOCKS_FN) );
}

void gpu_close(struct device_info *dev) {
	cutilDrvSafeCall( cuModuleUnload(dev->module)	);
  	cutilDrvSafeCall( cuCtxDestroy(dev->context) 		);
}

int merge_blocks(struct device_info *dev, CUdeviceptr *dataA, CUdeviceptr *dataB, 
											uint n_merge_blocks, uint merge_block_size) {
	int i, logBlocks;
	uint n_launch_threads, n_launch_blocks;

	
	/* There will be Log_2(blocks) merge steps. */
	logBlocks = 0;
	for (i=1; i<n_merge_blocks; i *= 2)	++logBlocks;

	if ( VERB & DEBUG ) printf("There will be %d merge steps.\n", logBlocks);	

	n_launch_threads = n_merge_blocks >> 1;  /* Start with blocks/2 threads */
	n_launch_blocks = (n_launch_threads-1) / MAX_THREADS_PER_BLOCK  +  1;

	cuParamSeti(dev->kernel2, 4, merge_block_size);
	cuParamSetSize(dev->kernel2, 16);

	for (i=0; i < logBlocks; ++i) {
		cuFuncSetBlockShape(dev->kernel2, n_launch_threads, 1, 1);
		cuParamSeti(dev->kernel2, 0, i); /* set merge level */

		/* Merging uses a source array and destination array, the gpu has 2 arrays allocated
		 * so we swap which is the source and which is the destination for each iteration. */
		if ( i%2 == 0 ) {
			cuParamSeti(dev->kernel2, 8, *dataA);
			cuParamSeti(dev->kernel2, 12, *dataB);
		} else {
			cuParamSeti(dev->kernel2, 8, *dataB);
			cuParamSeti(dev->kernel2, 12, *dataA);
		}

		/* Execute the kernel on the GPU */
		if ( VERB & DEBUG ) 
			printf("Launching block merge kernel with %d blocks and %d threads per block\n", 
														n_launch_blocks, n_launch_threads/n_launch_blocks);

		cutilDrvSafeCall( cuLaunchGrid(dev->kernel2, n_launch_blocks, 1) );

		/* We need half the number of threads for each iteration */
		n_launch_threads = n_launch_threads >> 1;	
		n_launch_blocks = (n_launch_threads-1) / MAX_THREADS_PER_BLOCK  +  1;
	}
	return i;
}

int D_bitonic_sort(int *data, uint length) {
	uint num_threads;
	uint num_blocks, block_size;
	uint nBytes;
	int status, i, logBlocks;

	struct device_info device_info;

	CUdeviceptr pDeviceArrayA;
	CUdeviceptr pDeviceArrayB;

	status = SUCCESS;

	/*
	* Phase 1:
	* 	There will be one thread for each element to be sorted. Each
	*	block will perform bitonic sort on MAX_THREADS_PER_BLOCK elements.
	*/

	/* Initialize sizes */
	num_threads = _min(length, MAX_THREADS_PER_BLOCK );
	num_blocks = (length-1) / MAX_THREADS_PER_BLOCK + 1;
	nBytes = length * sizeof(int);

	/* Initialize gpu */
	if (VERB & DEBUG) printf("Initializing GPU.\n");
	gpu_init(&device_info);

	/* Allocate memory on the device */
	cutilDrvSafeCall( cuMemAlloc(&pDeviceArrayA, nBytes)		);
	cutilDrvSafeCall( cuMemAlloc(&pDeviceArrayB, nBytes)		);
	cutilDrvSafeCall( cuMemcpyHtoD(pDeviceArrayA, data, nBytes) 	);

	/* Configure kernel function 1 */
	cutilDrvSafeCall( cuFuncSetBlockShape(device_info.kernel1, num_threads, 1, 1));
	cutilDrvSafeCall( cuParamSeti(device_info.kernel1, 0, pDeviceArrayA)	);
	cutilDrvSafeCall( cuParamSetSize(device_info.kernel1, 4)		);
	
	/* Execute the kernel on the GPU */
	if ( VERB & DEBUG ) 
		printf("Launching bitonic sort kernel with %d blocks and %d threads per block.\n", 
													num_blocks, num_threads/num_blocks);
	cutilDrvSafeCall( cuLaunchGrid(device_info.kernel1, num_blocks, 1)		);


	/*
	* Phase 2:
	* 	At this point each block is a sorted list. Now it's time to merge them.	
	*/

	/* Do we need to merge blocks? */
	if ( num_blocks > 1 ) {		
		i = merge_blocks(&device_info, &pDeviceArrayA, &pDeviceArrayB, num_blocks, num_threads);
	} else i=0;

	/* Determine if answer is in pDeviceArrayA or pDeviceArrayB and copy to host. */
	if (i%2 == 0) 	cutilDrvSafeCall( cuMemcpyDtoH(data, pDeviceArrayA, nBytes) );
	else			cutilDrvSafeCall( cuMemcpyDtoH(data, pDeviceArrayB, nBytes) );

	cutilDrvSafeCall( cuMemFree(pDeviceArrayA)	);
	cutilDrvSafeCall( cuMemFree(pDeviceArrayB)	);

	gpu_close(&device_info);	

	return i;
}
