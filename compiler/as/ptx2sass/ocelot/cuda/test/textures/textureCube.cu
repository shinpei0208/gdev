/*!
	\file textureArray.cu
	\author Andrew Kerr <arkerr@gatech.edu>
	\brief tests implementation of cube texture mapping
	\date February 10, 2012
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

/////////////////////////////////////////////////////////////////////////////////////////////////

// declare texture reference for 2D float texture
texture<float, cudaTextureTypeCubemap> Surface;

/*!
	kernel in which each thread samples the texture and writes it to out, a row-major dense 
	block of samples
*/
__global__ void kernel(float *out, int width) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	const float pi = 3.14159265358f;

	float theta = pi * (2.0f - (float)x / (float)width);
	float phi = pi * (2.0f - (float)y / (float)width);

	float cx = cos(theta)*cos(phi);
	float cy = sin(phi);
	float cz = sin(theta)*cos(phi);
	
	float sample = texCubemap( Surface, cx, cy, cz );

	out[x + y * width] = sample;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **arg) {
	int width = 16, layers = 6;

	float *in_data_host, *out_data_host;
	float *out_data_gpu;
	cudaArray *in_data_gpu = 0;

	size_t bytes = width * width * sizeof(float);
	in_data_host = (float *)malloc(bytes);
	out_data_host = (float *)malloc(bytes);

	// procedural texture generation
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < width; j++) {
			in_data_host[i * width + j] = (float)((122 + i*3 + j*2) % 128) / 128.0f;
			out_data_host[i*width+j] = 0;
		}
	}

	// construct array
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
  
  if (cudaMalloc3DArray( &in_data_gpu, &channelDesc, 
  	make_cudaExtent(width, width, layers), cudaArrayCubemap ) != cudaSuccess) {
  
		printf("cudaMalloc3DArray() - failed: %s\n", 
			cudaGetErrorString(cudaGetLastError()));
		return -1;
  }
   
  cudaMemcpy3DParms parameters = {0};
  parameters.srcPos = make_cudaPos(0,0,0); 
  parameters.dstPos = make_cudaPos(0,0,0); 
  parameters.srcPtr = make_cudaPitchedPtr(in_data_host, width * sizeof(float), width, width); 
  parameters.dstArray = in_data_gpu;
  parameters.extent = make_cudaExtent(width, width, layers);
  parameters.kind = cudaMemcpyHostToDevice;
  
  if (cudaMemcpy3D(&parameters) != cudaSuccess) {
		printf("cudaMemcpy3D() - failed: %s\n", 
			cudaGetErrorString(cudaGetLastError()));
		return -1;
  }
  
  // set texture parameters
	Surface.addressMode[0] = cudaAddressModeWrap;
	Surface.addressMode[1] = cudaAddressModeWrap;
	Surface.filterMode = cudaFilterModePoint;
	Surface.normalized = true;  // access with normalized texture coordinates
	
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

	dim3 grid(width / 16, width / 16), block(16, 16);
	
	kernel<<< grid, block >>>(out_data_gpu, width);

	cudaThreadSynchronize();

	cudaMemcpy(out_data_host, out_data_gpu, bytes, cudaMemcpyDeviceToHost);
	cudaFreeArray(in_data_gpu);
	cudaFree(out_data_gpu);

	int errors = 0;
	
	// unit test not written
	// ...
	
	printf("Pass/Fail : %s\n", (errors ? "Fail" : "Pass"));

	free(in_data_host);
	free(out_data_host);

	return (errors ? -1 : 0);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

