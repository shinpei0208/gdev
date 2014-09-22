/*!
	\file indirectCallDriver.cpp
	
	\author Andrew Kerr <arkerr@gatech.edu>
	
	\brief demonstrates indirect function calling
*/


#include <cstdio>
#include <ocelot/cuda/interface/cuda.h>

int main(int argc, char *arg[]) {

	const int P = 0;
	const int N = 32;
	int *A_cpu;
	size_t bytes = sizeof(int) * N;
	
	CUdeviceptr A_gpu;
	
	CUresult result;
	CUdevice device;
	CUcontext context;
	CUmodule module;
	CUfunction function;
	
	result = cuInit(0);
	if (result != CUDA_SUCCESS) {
		printf("cuInit() failed: %d\n", (int)result);
		return 1;
	}
	
	result = cuDeviceGet(&device, 0);
	if (result != CUDA_SUCCESS) {
		printf("cuDeviceGet() failed: %d\n", (int)result);
		return 1;
	}
	
	result = cuCtxCreate(&context, CU_CTX_SCHED_AUTO, device);
	if (result != CUDA_SUCCESS) {
		printf("cuCtxCreate() failed: %d\n", (int)result);
		return 1;
	}
	
	result = cuModuleLoad(&module, "ocelot/cuda/test/functions/indirectCallDriver.ptx");
	if (result != CUDA_SUCCESS) {
		printf("cuModuleLoad() failed: %d\n", (int)result);
		return 1;
	}
	
	result = cuModuleGetFunction(&function, module, "kernelEntry");
	if (result != CUDA_SUCCESS) {
		printf("cuModuleLoad() failed: %d\n", (int)result);
		return 1;
	}
	
	result = cuMemAlloc(&A_gpu, bytes);
	if (result != CUDA_SUCCESS) {
		printf("cuMemAlloc() failed: %d\n", (int)result);
		return 1;
	}
	
	A_cpu = (int *)malloc(bytes);
	for (int i = 0; i < N; i++) {
		A_cpu[i] = 0;
	}
	
	result = cuMemcpyHtoD(A_gpu, A_cpu, bytes);
	if (result != CUDA_SUCCESS) {
		printf("cuMemcpyHtoD() failed: %d\n", (int)result);
		return 1;
	}
		
	result = cuFuncSetBlockShape(function, 32, 1, 1);
	if (result != CUDA_SUCCESS) {
		printf("cuFuncSetBlockShape() failed: %d\n", (int)result);
		return 1;
	}
	
	struct {
		void *A;
		int P;
	} kernelEntryParams;
	
	result = cuParamSetSize(function, sizeof(void *) + sizeof(int));
	if (result != CUDA_SUCCESS) {
		printf("cuParamSetSize() failed: %d\n", (int)result);
		return 1;
	}
	
	kernelEntryParams.A = (void *)A_gpu;
	kernelEntryParams.P = P;
	
	result = cuParamSetv(function, 0, &kernelEntryParams.A, sizeof(void *));
	if (result != CUDA_SUCCESS) {
		printf("cuParamSetv() failed: %d\n", (int)result);
		return 1;
	}
	result = cuParamSetv(function, sizeof(void *), &kernelEntryParams.P, sizeof(int));
	if (result != CUDA_SUCCESS) {
		printf("cuParamSetv() failed: %d\n", (int)result);
		return 1;
	}
	
	result = cuLaunchGrid(function, (N + 32 - 1) / 32, 1);
	if (result != CUDA_SUCCESS) {
		printf("cuLaunchGrid() failed: %d\n", (int)result);
		return 1;
	}
	
	result = cuCtxSynchronize();
	if (result != CUDA_SUCCESS) {
		printf("cuCtxSynchronize() failed: %d\n", (int)result);
		return 1;
	}
	
	result = cuMemcpyDtoH(A_cpu, A_gpu, bytes);
	if (result != CUDA_SUCCESS) {
		printf("cuMemcpyDtoH() failed: %d\n", (int)result);
		return 1;
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

