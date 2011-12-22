#include <cuda.h>
#include "nn.h"

/**
 * Kernel
 * Executed on GPU
 * Calculates the Euclidean distance from each record in the database to the 
 * target position
 */
__global__ void euclid
(LatLong *d_locations, float *d_distances, int numRecords,float lat, float lng)
{
	int globalId = blockIdx.x * blockDim.x + threadIdx.x;
	LatLong *latLong = d_locations + globalId;
	if (globalId < numRecords) {
	   float *dist = d_distances + globalId;
	   *dist = (float)sqrt((lat-latLong->lat) * (lat-latLong->lat) + (lng - latLong->lng) * (lng - latLong->lng));
	}
}
