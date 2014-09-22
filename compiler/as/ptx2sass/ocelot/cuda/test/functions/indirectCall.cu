/*!
	\file indirectCall.cu
	
	\author Andrew Kerr <arkerr@gatech.edu>
	
	\brief demonstrates indirect function calling
*/

extern "C" __device__ __noinline__ int funcDouble(int a) {
	return a*2;
}

extern "C" __device__ __noinline__ int funcTriple(int a) {
	return a*3;
}

extern "C" __device__ __noinline__ int funcQuadruple(int a) {
	return a*4;
}
extern "C" __device__ __noinline__ int funcPentuple(int a) {
	return a*5;
}

extern "C" __global__ void kernelEntry(int *A, int b) {

	/*
	int (*filter[])(int) = {
		&funcDouble,
		&funcTriple,
		&funcQuadruple,
		&funcPentuple
	};
	*/
	
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int p = ((b + i) & 3);
	
	int (*filter)(int);
	if (p == 0) {
		filter = &funcDouble;
	}
	else if (p == 1) {
		filter = &funcTriple;
	}
	else if (p == 2) {
		filter = &funcQuadruple;
	}
	else if (p == 3) {
		filter = &funcPentuple;
	}

	A[i] = filter(i);
}

#include <cstdio>

int main(int argc, char *arg[]) {

	const int P = 0;
	const int N = 32;
	int *A_gpu, *A_cpu;
	size_t bytes = sizeof(int) * N;
	
	cudaError_t result = cudaThreadSynchronize();
	if (result != cudaSuccess) {
		printf("Initialization error:%s\n", cudaGetErrorString(result));
		return 1;
	}
	
	result = cudaMalloc((void **)&A_gpu, bytes);
	if (result != cudaSuccess) {
		printf("cudaMalloc() - failed to allocate %d on the device \n", (int)bytes);
		return 2;
	}
	
	A_cpu = (int *)malloc(bytes);
	for (int i = 0; i < N; i++) {
		A_cpu[i] = 0;
	}
	
	result = cudaMemcpy(A_gpu, A_cpu, bytes, cudaMemcpyHostToDevice);
	if (result != cudaSuccess) {
		printf("cudaMemcpy() - failed to copy %d bytes TO the device\n", (int)bytes);
		return 2;
	}
	
	dim3 block(32, 1);
	dim3 grid((N + block.x - 1) / block.x, 1);
	
	kernelEntry<<< grid, block >>>(A_gpu, P);
	
	result = cudaThreadSynchronize();
	if (result != cudaSuccess) {
		printf("Kernel launch error: %s\n", cudaGetErrorString(result));
		return 3;
	}
	
	result = cudaMemcpy(A_cpu, A_gpu, bytes, cudaMemcpyDeviceToHost);
	if (result != cudaSuccess) {
		printf("cudaMemcpy() - failed to copy %d bytes FROM the device\n", (int)bytes);
		return 2;
	}
	
	int errors = 0;
	for (int i = 0; i < N; i++) {
		int got = A_cpu[i];
		int dem = 0;
		
		int p = ((P + i) % 4);
		dem = ((p + 2) * i);
		
		if (got != dem) {
			printf("Error[%d] - expected: %d, got: %d\n", i, dem, got);
			if (++errors > 5) {
				break;
			}
		}
	}
	
	cudaFree(A_gpu);
	free(A_cpu);
	
	if (errors) {
		printf("FAILED\n");
		printf(" with %d errors\n", errors);
	}
	else {
		printf("Pass/Fail : Pass\n");
	}
	
	return 0;
}

