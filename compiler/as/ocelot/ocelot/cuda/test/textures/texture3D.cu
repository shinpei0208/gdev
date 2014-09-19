/*!
	\file texture3D.cu

	\author Andrew Kerr <arkerr@gatech.edu>

	\brief tests implementation of cudaBindTextureToArray

	\date 27 Oct 2009
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

/////////////////////////////////////////////////////////////////////////////////////////////////

// declare texture reference for 2D float texture
texture<float, 3, cudaReadModeElementType> Surface;

extern "C" __global__ void kernelMemset(float *out, int width, int height, float value) {
	unsigned int x = threadIdx.x;
	unsigned int y = threadIdx.y;
	unsigned int z = blockIdx.x;
	out[x + y * width + z * width * height] = value;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/*!
	kernel in which each thread samples the texture and writes it to out, a row-major dense 
	block of samples
	
*/
extern "C" __global__ void kernel(float *out, int width, int height) {

	unsigned int x = threadIdx.x;
	unsigned int y = threadIdx.y;
	unsigned int z = blockIdx.x;

	float sample = tex3D(Surface, x, y, z);

	out[x + y * width + z * width * height] = sample;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **arg) {
	int width = 8, height = 8, depth = 4;

	float *in_data_host, *out_data_host;
	float *out_data_gpu;

	cudaError_t result;
	cudaArray *arrayPointer;
	size_t bytes = width * height * depth * sizeof(float);
	in_data_host = (float *)malloc(bytes);
	out_data_host = (float *)malloc(bytes);

	// initialize data
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	
	cudaExtent extent;
	extent.width = width;
	extent.depth = depth;
	extent.height = height;

	result = cudaMalloc3DArray(&arrayPointer, &channelDesc, extent, 0);
	if (result != cudaSuccess) {
		fprintf(stderr, "Texture3D - failed to malloc 3D array - %s", cudaGetErrorString(result));
		return 1;
	}
	
	// memcpy
	for (int k = 0; k <depth; k++) {
		for (int j = 0; j < height; j++) {
			for (int i = 0; i < width; i++) {
				float c = 0;
				
				// impulse response
				if (i == 4 && j == 4 && k == 1) { c = 1.0f; }
				
				in_data_host[i + j * width + k * width * height] = c;
			}
		}
	}
	
	cudaMemcpy3DParms params;

	memset(&params, 0, sizeof(params));
	params.srcPtr.pitch = sizeof(float) * width;
	params.srcPtr.ptr = in_data_host;
	params.srcPtr.xsize = width;
	params.srcPtr.ysize = height;

	params.srcPos.x = 0;
	params.srcPos.y = 0;
	params.srcPos.z = 0;

	params.dstArray = arrayPointer;

	params.dstPos.x = 0;
	params.dstPos.y = 0;
	params.dstPos.z = 0;

	params.extent.width = width;
	params.extent.depth = depth;
	params.extent.height = height;

	params.kind = cudaMemcpyHostToDevice;

	result = cudaMemcpy3D(&params);
	if (result != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy3D - failed to copy from host buffer to device array - %s\n", cudaGetErrorString(result));
		return 1;
	}
	
	// clear it, copy back from array
	for (int k = 0; k <depth; k++) {
		for (int j = 0; j < height; j++) {
			for (int i = 0; i < width; i++) {
				in_data_host[i + j * width + k * width * height] = -1;
			}
		}
	}
	
	memset(&params, 0, sizeof(params));
	params.dstPtr.pitch = sizeof(float) * width;
	params.dstPtr.ptr = in_data_host;
	params.dstPtr.xsize = width;
	params.dstPtr.ysize = height;

	params.srcPos.x = 0;
	params.srcPos.y = 0;
	params.srcPos.z = 0;

	params.srcArray = arrayPointer;

	params.dstPos.x = 0;
	params.dstPos.y = 0;
	params.dstPos.z = 0;

	params.extent.width = width;
	params.extent.depth = depth;
	params.extent.height = height;

	params.kind = cudaMemcpyDeviceToHost;

	result = cudaMemcpy3D(&params);
	if (result != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy3D - failed to copy from array to host buffer for verification - %s", cudaGetErrorString(result));
		return 1;
	}
	for (int k = 0; k <depth; k++) {
		for (int j = 0; j < height; j++) {
			for (int i = 0; i < width; i++) {
				float c = in_data_host[i + j * width + k * width * height];
				if (c < 0) {
					fprintf(stderr, "You can't copy to and from a 3D array. What makes you think you can sample from one? (%d, %d, %d)\n",
						i, j, k);
					fprintf(stderr, " = %f\n", c);
					return 1;
				}
			}
		}
	}

	Surface.addressMode[0] = cudaAddressModeWrap;
	Surface.addressMode[1] = cudaAddressModeWrap;
	Surface.addressMode[2] = cudaAddressModeWrap;
	Surface.filterMode = cudaFilterModeLinear;
	Surface.normalized = false;
	
	// bind to array
	result = cudaBindTextureToArray(Surface, arrayPointer, channelDesc);
	if (result != cudaSuccess) {
		fprintf(stderr, "cudaBindTextureToArray() - failed to bind texture to array - %s", cudaGetErrorString(result));
		return 2;
	}
	
	// allocate output
	if (cudaMalloc((void **)&out_data_gpu, bytes) != cudaSuccess) {
		printf("cudaMalloc(out_data_gpu) - failed to allocate %d bytes: %s\n", (int)bytes,
			cudaGetErrorString(cudaGetLastError()));
		return -2;
	}
	
	dim3 grid(depth, 1);
	dim3 block(width, height, 1);
	kernelMemset<<< grid, block >>>(out_data_gpu, width, height, 0.0f);
	kernel<<< grid, block >>>(out_data_gpu, width, height);
	cudaThreadSynchronize();
	
	result = cudaMemcpy(out_data_host, out_data_gpu, bytes, cudaMemcpyDeviceToHost);
	if (result != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy() - failed to copy from device output buffer to host buffer %s", cudaGetErrorString(result));
		return 3;
	}
	
	// print result as a set of matrices
	for (int k = 0; k < depth; k++) {
		float *ptr = &out_data_host[k * width * height];
		
		printf("\nA_%d = [\n", k);
		for (int j = 0; j < height; j++) {
			for (int i = 0; i < width; i++) {
				printf(" %f ", ptr[i + j * width]);
			}
			printf(";\n");
		}
		printf("];\n");
	}
	
	// clean up
	cudaFree(out_data_gpu);
	free(out_data_host);
	free(in_data_host);
	cudaFreeArray(arrayPointer);
	
	return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

