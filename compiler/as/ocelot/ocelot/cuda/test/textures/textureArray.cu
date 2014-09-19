/*!
	\file textureArray.cu

	\author Andrew Kerr <arkerr@gatech.edu>

	\brief tests implementation of cudaBindTextureArray

	\date 27 Oct 2009
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

/////////////////////////////////////////////////////////////////////////////////////////////////

// declare texture reference for 2D float texture
texture<float, 2, cudaReadModeElementType> Surface;

/*!
	kernel in which each thread samples the texture and writes it to out, a row-major dense 
	block of samples
*/
__global__ void kernel(float *out, int width, int height) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	float sample = tex2D(Surface, x, y);

	out[x + y * width] = sample;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **arg) {
	int width = 64, height = 64;

	float *in_data_host, *out_data_host;
	float *out_data_gpu;
	cudaArray *in_data_gpu = 0;

	size_t bytes = width * height * sizeof(float);
	in_data_host = (float *)malloc(bytes);
	out_data_host = (float *)malloc(bytes);

	// procedural texture generation
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			in_data_host[i * width + j] = (float)((122 + i*3 + j*2) % 128) / 128.0f;
			out_data_host[i*width+j] = 0;
		}
	}

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	if (cudaMallocArray(&in_data_gpu, &channelDesc, width, height) != cudaSuccess) {
		printf("failed to malloc array: %s\n", cudaGetErrorString(cudaGetLastError()));
		free(in_data_host);
		free(out_data_host);
		return -2;
	}
	
	if (cudaMemcpyToArray(in_data_gpu, 0, 0, in_data_host, bytes, cudaMemcpyHostToDevice)
		!= cudaSuccess) {
		
		printf("cudaMemcpyToArray() - failed to bind texture: %s\n", 
			cudaGetErrorString(cudaGetLastError()));
		
		free(in_data_host);
		free(out_data_host);
		cudaFreeArray(in_data_gpu);
		return -2;		
	}

	Surface.addressMode[0] = cudaAddressModeWrap;
	Surface.addressMode[1] = cudaAddressModeWrap;
	Surface.filterMode = cudaFilterModePoint;
	Surface.normalized = false;
	
	if (cudaBindTextureToArray(Surface, in_data_gpu, channelDesc) != cudaSuccess) {
		
		printf("cudaBindTextureToArray() - failed to bind texture: %s\n", 
			cudaGetErrorString(cudaGetLastError()));
		
		free(in_data_host);
		free(out_data_host);
		cudaFreeArray(in_data_gpu);
		return -2;
	}

	if (cudaMalloc((void **)&out_data_gpu, bytes) != cudaSuccess) {
		
		printf("cudaMalloc(out_data_gpu) - failed to allocate %d bytes: %s\n", (int)bytes,
			cudaGetErrorString(cudaGetLastError()));
		
		free(in_data_host);
		free(out_data_host);
		cudaFreeArray(in_data_gpu);
		return -2;
	}

	dim3 grid(width / 16, height / 16), block(16, 16);
	
	kernel<<< grid, block >>>(out_data_gpu, width, height);

	cudaThreadSynchronize();

	cudaMemcpy(out_data_host, out_data_gpu, bytes, cudaMemcpyDeviceToHost);
	cudaFreeArray(in_data_gpu);
	cudaFree(out_data_gpu);

	int errors = 0;
	for (int i = 0; i < height && errors < 5; i++) {
		for (int j = 0; j < width && errors < 5; j++) {
			float in = in_data_host[i * width + j];
			float out = out_data_host[i * width + j];
			if (fabs(in - out) > 0.001f) {
				++errors;
				printf("(%d, %d) - in = %f, out = %f %s\n", i, j, in, out, (errors ? "***":""));
			}
		}
	}

	printf("Pass/Fail : %s\n", (errors ? "Fail" : "Pass"));

	free(in_data_host);
	free(out_data_host);

	return (errors ? -1 : 0);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

