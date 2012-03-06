/*****************************************************************************
* Name        : D_bitonic_sort.h
* Author      : Jason Aumiller
* Version     :
* Copyright   :  
* Description : CUDA Kernel function for bitonic sort
/*****************************************************************************/


#include <stdio.h>


__device__ void swap(uint a, uint b, int *data){
	uint temp = data[a];
	data[a]=data[b];
	data[b]=temp;
}

__global__ void D_bitonic_sort_K(int *data_to_sort)
{
	uint desc; // bit position that determines sort order ascending/descending
	uint stride; // distance to partner index
	int thread = threadIdx.x;

	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	for (desc=2; desc<=blockDim.x; desc*=2) {
		for (stride = desc>>1; stride>0; stride=stride>>1) {
			int partner = idx ^ stride;
			__syncthreads();
			if (partner > idx) {
				if ( (thread & desc) == 0 && data_to_sort[idx] > data_to_sort[partner] ) {
					swap(idx,partner,data_to_sort);
				} else if ( (thread & desc) != 0 && data_to_sort[idx]< data_to_sort[partner] ) {
					swap(idx,partner,data_to_sort);
				}
			}
		}
	}
}

__global__ void D_merge_blocks_K(uint merge_level, uint block_size, int *data_in, int *data_out) {

	//const uint block 	= (uint) blockIdx.x;

	const uint start 	= block_size * ( threadIdx.x << (merge_level + 1) );
	const uint range	= block_size << merge_level;

	int i;
	int da, db, end;
	da = start;
	db = start + range;
	end = start + 2*range;

	for (i=start; i < end; ++i){
		if (da >= start + range) {
			data_out[i] = data_in[db++];
		} else if (db >= end) {
			data_out[i] = data_in[da++];
		} else if ( data_in[da] <= data_in[db] ) {
			data_out[i] = data_in[da++];
		} else {
			data_out[i] = data_in[db++];
		}
	}
}

