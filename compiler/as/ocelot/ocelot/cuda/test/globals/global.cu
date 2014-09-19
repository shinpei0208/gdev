/*!
	\file global.cu
	\author Andrew Kerr <arkerr@gatech.edu>
	\brief verifies a CUDA application's ability to use global symbols
	\date Feburary 12, 2010
*/

#include <stdio.h>

__device__ float Pi;

extern "C" __global__ void copyFromGlobal(float *result) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	result[i] = Pi * (float)(i % 128);
}

int main(int argc, char *arg[]) {
	int N = 64;
	bool verbose = false;
	size_t bytes = sizeof(float) * N;
	float *results_gpu = 0;
	float *results_cpu = (float *)malloc(bytes);
	int devices = 0;
	cudaGetDeviceCount(&devices);
	
	int errors = 0;
	for (int device = 0; device != devices; ++device) {
		cudaDeviceProp properties;
		cudaGetDeviceProperties(&properties, device);
		printf("cudaSetDevice() - %d - %s \n", device, properties.name);
		cudaSetDevice(device);
		errors = 0;
		
		if (cudaMalloc((void **)&results_gpu, bytes) != cudaSuccess) {
			printf("cudaMalloc() failed to allocate %d bytes on device\n", (int)bytes);
			return -1;
		}

		for (int i = 0; i < N; i++) {
			results_cpu[i] = -1;
		}
		if (verbose) {
			printf(" [1]\n");
		}
		cudaMemcpy(results_gpu, results_cpu, bytes, cudaMemcpyHostToDevice);
	
		if (verbose) {
			printf(" [2]\n");
		}
	
		float pi = 3.14159f;
		if (cudaMemcpyToSymbol("Pi", &pi, sizeof(float), 0, 
			cudaMemcpyHostToDevice) != cudaSuccess) {
		
			printf("cudaMemcpyToSymbol() failed to copy 4 bytes to symbol 'Pi'\n");
		
			cudaFree(results_gpu);
			free(results_cpu);
			return -1;
		}
	
	
		if (verbose) {
			printf(" [3]\n");
		}
		float copy_pi = 0;
		if (cudaMemcpyFromSymbol(&copy_pi, "Pi", sizeof(float), 0,
			cudaMemcpyDeviceToHost) != cudaSuccess) {
		
			printf("cudaMemcpyFromSymbol() failed to copy 4 bytes from symbol 'Pi'\n");
		
			cudaFree(results_gpu);
			free(results_cpu);
			return -1;
		}
	
		if (fabs(copy_pi - 3.14159f) > 0.001f) {
			printf("value copied from symbol (%f) did not match expected 3.14159\n",
				copy_pi);
		
			cudaFree(results_gpu);
			free(results_cpu);
			return -1;
		}
	
		dim3 block(64, 1);
		dim3 grid((63 + block.x) / 64, 1);
	
		copyFromGlobal<<< grid, block >>>(results_gpu);
	
		if (verbose) {
			printf(" [4]\n");
		}
		cudaMemcpy(results_cpu, results_gpu, bytes, cudaMemcpyDeviceToHost);
	
		for (int i = 0; i < N; i++) {
			float expected = 3.14159f * (float)(i % 128);
			float got = results_cpu[i];
			if (fabs(expected - got) > 0.001f) {
				printf("ERROR 0 - [%d] - got: %f, expected: %f\n", i, got, expected);
				if (++errors > 5) { break; }
			}
		}
	
		if (verbose) {
			printf("[5]\n");
		}
		float *pi_gpu = 0;
		if (cudaGetSymbolAddress((void **)&pi_gpu, "Pi") != cudaSuccess) {
			printf("failed to get address of global variable 'Pi'\n");
			cudaFree(results_gpu);
			free(results_cpu);
			return -1;
		}
	
		if (verbose) {
			printf(" [6]\n");
		}
		copy_pi = 2.0f * 3.14159f;
		if (cudaMemcpy(pi_gpu, &copy_pi, sizeof(float), cudaMemcpyHostToDevice) !=
			cudaSuccess) {
		
			printf("failed to copy value to symbol 'Pi'\n");
			cudaFree(results_gpu);
			free(results_cpu);
			return -1;		
		}

		copyFromGlobal<<< grid, block >>>(results_gpu);
	
		if (verbose) {
			printf(" [7]\n");
		}
		cudaMemcpy(results_cpu, results_gpu, bytes, cudaMemcpyDeviceToHost);
	
		for (int i = 0; i < N; i++) {
			float expected = 2.0f * 3.14159f * (float)(i % 128);
			float got = results_cpu[i];
			if (fabs(expected - got) > 0.001f) {
				printf("ERROR 1 - [%d] - got: %f, expected: %f\n",
					i, got, expected);
				if (++errors > 5) { break; }
			}
		}
	
		cudaFree(results_gpu);

	}
	
	printf("Pass/Fail : %s\n", (errors ? "Fail" : "Pass"));

	free(results_cpu);

	return 0;
}

