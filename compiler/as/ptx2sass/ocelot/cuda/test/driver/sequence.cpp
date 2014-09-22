/*!
	\file sequence.cpp
	\brief CUDA Driver API version of the sequence program
*/

#include <ocelot/cuda/interface/cuda.h>
//#include <cuda.h>

#include <iostream>

#define report(x) std::cout << x << std::endl

std::ostream & operator<<(std::ostream &out, CUresult r) {
	out << "CUDA ERROR: " << (int)r;
	return out;
}

int main() {
	const int N = 256;
	
	int bytes = sizeof(int) * N;
	int *A_cpu = 0;
	CUdeviceptr A_gpu = 0;
	
	A_cpu = (int *)malloc(bytes);
	for (int i = 0; i < N; i++) {
		A_cpu[i] = -1;
	}
	
	CUresult result = cuInit(0);
	if (result != CUDA_SUCCESS) {
		report("cuInit() failed: " << result);
		return 1;
	}
	
	int driverVersion = 0;
	result = cuDriverGetVersion(&driverVersion);
	if (result != CUDA_SUCCESS) {
		report("cuDriverGetVersion() failed: " << result);
	}
	report("cuda driver version: " << driverVersion);
	
	int count = 0;
	result = cuDeviceGetCount(&count);
	if (result != CUDA_SUCCESS) {
		report("cuDeviceGetCount() failed: " << result);
		return 1;
	}
	report("  " << count << " devices");
	
	CUdevice device;
	result = cuDeviceGet(&device, 0);
	if (result != CUDA_SUCCESS) {
		report("cuDeviceGet() failed: " << result);
		return 1;
	}
	
	char devName[256] = {0};
	result = cuDeviceGetName(devName, 255, device);
	if (result != CUDA_SUCCESS) {
		report("cuDeviceGetName() failed: " << result);
		return 1;
	}
	report("using device '" << devName << "'");
	
	int major, minor;
	result = cuDeviceComputeCapability(&major, &minor, device);
	if (result != CUDA_SUCCESS) {
		report("cuDeviceComputeCapability() failed: " << result);
		return 1;
	}
	report("  compute capability " << major << "." << minor);
	
	CUcontext ctx;
	CUmodule module;
	CUfunction function;

	result = cuCtxCreate(&ctx, 0, device);
	if (result != CUDA_SUCCESS) {
		report("cuCtxCreate() failed: " << result);
		return 1;
	}
		
	int pi = 0;
	result = cuDeviceGetAttribute(&pi, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, device);
	if (result != CUDA_SUCCESS) {
		report("cuDeviceGetAttribute() failed: " << result);
	}
	report("  compute mode: " << pi);
	
	result = cuModuleLoad(&module, "ocelot/cuda/test/driver/sequence.ptx");
	if (result != CUDA_SUCCESS) {
		report("cuModuleLoad() failed: " << result);
		return 1;
	}
	
	result = cuModuleGetFunction(&function, module, "sequence");
	if (result != CUDA_SUCCESS) {
		report("cuModuleGetFunction() failed: " << result);
		return 1;
	}
	
	result = cuMemAlloc(&A_gpu, bytes);
	if (result != CUDA_SUCCESS) {
		report("cuMemAlloc() failed: " << result);
		return 1;
	}
	
	result = cuMemcpyHtoD(A_gpu, A_cpu, bytes);
	if (result != CUDA_SUCCESS) {
		report("cuMemcpyHtoD() failed: " << result);
		return 1;
	}
	
	struct  {
		int *A;
		int N;
	} parameters;
	
	result = cuParamSetSize(function, sizeof(parameters));
	if (result != CUDA_SUCCESS) {
		report("cuParamSetSize() failed: " << result);
		return 1;
	}
	parameters.A = reinterpret_cast<int *>(A_gpu);
	parameters.N = N;
	result = cuParamSetv(function, 0, &parameters.A, sizeof(parameters.A));
	if (result != CUDA_SUCCESS) {
		report("cuParamSetv() failed: " << result);
		return 1;
	}
	result = cuParamSetv(function, 8, &parameters.N, sizeof(parameters.N));
	if (result != CUDA_SUCCESS) {
		report("cuParamSetv() failed: " << result);
		return 1;
	}
	
	int gridWidth = (N + 63) / 64;	
	result = cuFuncSetBlockShape(function, 64, 1, 1);
	if (result != CUDA_SUCCESS) {
		report("cuFuncSetBlockShape() failed: " << result);
		return 1;
	}
	result = cuLaunchGrid(function, gridWidth, 1);
	if (result != CUDA_SUCCESS) {
		report("cuLaunchGrid() failed: " << result);
		return 1;
	}
	
	result = cuMemcpyDtoH(A_cpu, A_gpu, bytes);
	if (result != CUDA_SUCCESS) {
		report("cuMemcpyDtoH() failed: " << result);
		return 1;
	}
	
	cuModuleUnload(module);
	cuCtxDestroy(ctx);
	
	int errors = 0;
	for (int i = 0; i < N; i++) {
		int expected = i + 1;
		int got = A_cpu[i];
		if (expected != got) {
			report("ERROR [" << i << "] - expected: " << expected << ", got: " << got);
			report("  A_gpu[" << i << "] = " << ((int *)A_gpu)[i]);
			++errors;
			if (errors >= 10) {
				break;
			}
		}
	}
	
	free(A_cpu);
	
	if (errors) {
		report("FAILED with " << errors << " errors");
	}
	else {
		report("pass/fail: Passed");
	}
	
	return 0;
}

