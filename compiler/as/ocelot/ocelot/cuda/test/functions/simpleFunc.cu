/*!
	\file simpleFunc.cu
	\author Andrew Kerr <arkerr@gatech.edu>
	\brief demonstrates function calls in CUDA - this thoroughly chokes Ocelot
*/

#include <stdio.h>

extern "C"  __noinline__ __device__ float square(float f) {
	return f * f;
}

extern "C" __global__ void kernel(float *A, int N) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;	
	if (i < N) {
		A[i] = square(A[i]);
	}
}

int main() {
	float *A_gpu, *A_cpu;
	const int N = 32;
	size_t bytes = sizeof(float)*N;
	int errors = 0;
	
	A_cpu = (float *)malloc(bytes);
	cudaMalloc((void **)&A_gpu, bytes);
	
	dim3 block(32, 1);
	dim3 grid((N + 31) / 32, 1);
	
	for (int i = 0; i < N; i++) {
		A_cpu[i] = (float)i;
	}
	cudaMemcpy(A_gpu, A_cpu, bytes, cudaMemcpyHostToDevice);
	kernel<<< grid, block >>>(A_gpu, N);
	cudaThreadSynchronize();
	cudaMemcpy(A_cpu, A_gpu, bytes, cudaMemcpyDeviceToHost);
	
	for (int i = 0; i < N; i++) {
		float got = A_cpu[i];
		float expected = (float)i * (float)i;
		if (std::fabs(got - expected) > 0.001) {
			++errors;
			if (errors >= 5) {
				printf("Error[ %d ] - expected %f, got %f\n", i, expected, got);
				break;
			}
		}
	}
	
	free(A_cpu);
	cudaFree(A_gpu);
	
	printf("Pass/Fail : %s\n", (errors ? "Fail": "Pass"));
	
	return 0;
}

