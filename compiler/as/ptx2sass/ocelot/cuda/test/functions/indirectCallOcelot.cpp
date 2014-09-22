/*!
	\file indirectCall.cu
	
	\author Andrew Kerr <arkerr@gatech.edu>
	
	\brief demonstrates indirect function calling
*/


#include <cstdio>
#include <fstream>

#include <ocelot/cuda/interface/cuda_runtime.h>
#include <ocelot/api/interface/ocelot.h>

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
		printf("cudaMalloc() - failed to allocate %d on the device \n",
			(int)bytes);
		return 2;
	}
	
	A_cpu = (int *)malloc(bytes);
	for (int i = 0; i < N; i++) {
		A_cpu[i] = 0;
	}
	
	result = cudaMemcpy(A_gpu, A_cpu, bytes, cudaMemcpyHostToDevice);
	if (result != cudaSuccess) {
		printf("cudaMemcpy() - failed to copy %d bytes TO the device\n",
			(int)bytes);
		return 2;
	}
	
	std::ifstream file("ocelot/cuda/test/functions/indirectCall.ptx");
	ocelot::registerPTXModule(file, "indirectCall.cu");
	
	dim3 block(32, 1);
	dim3 grid((N + block.x - 1) / block.x, 1);
	cudaConfigureCall(grid, block);
	cudaSetupArgument(&A_gpu, sizeof(A_gpu), 0);
	cudaSetupArgument(&P, sizeof(P), sizeof(A_gpu));
	ocelot::launch("indirectCall.cu", "kernelEntry");
	
	result = cudaThreadSynchronize();
	if (result != cudaSuccess) {
		printf("Kernel launch error: %s\n", cudaGetErrorString(result));
		return 3;
	}
	
	result = cudaMemcpy(A_cpu, A_gpu, bytes, cudaMemcpyDeviceToHost);
	if (result != cudaSuccess) {
		printf("cudaMemcpy() - failed to copy %d bytes FROM the device\n",
			(int)bytes);
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
		printf("Pass/Fail: Pass\n");
	}
	
	return 0;
}

