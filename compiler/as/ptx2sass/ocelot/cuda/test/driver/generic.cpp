/*!
	\file genericmemory.cpp
	\brief CUDA Driver API version of the sequence program
*/

#include <ocelot/cuda/interface/cuda.h>
#include <iostream>

#define report(x) std::cout << x << std::endl

std::ostream & operator<<(std::ostream &out, CUresult r) {
	out << "CUDA ERROR: " << (int)r;
	return out;
}

int main() {
	const int N = 9;
	
	int bytes = sizeof(int) * N;
	int *A_cpu = 0;
	CUdeviceptr A_gpu = 0;
	
	A_cpu = new int[N];
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
	
	int count = 0;
	result = cuDeviceGetCount(&count);
	if (result != CUDA_SUCCESS) {
		report("cuDeviceGetCount() failed: " << result);
		return 1;
	}
	
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
	
	int major, minor;
	result = cuDeviceComputeCapability(&major, &minor, device);
	if (result != CUDA_SUCCESS) {
		report("cuDeviceComputeCapability() failed: " << result);
		return 1;
	}
	
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
	
	result = cuModuleLoad(&module, "ocelot/cuda/test/driver/generic.ptx");
	if (result != CUDA_SUCCESS) {
		report("cuModuleLoad() failed: " << result);
		return 1;
	}
	
	result = cuModuleGetFunction(&function, module, "genericmemory");
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
	
	struct {
		int *A;
	} parameters;
	
	result = cuParamSetSize(function, sizeof(parameters));
	if (result != CUDA_SUCCESS) {
		report("cuParamSetSize() failed: " << result);
		return 1;
	}
	parameters.A = reinterpret_cast<int *>(A_gpu);
	result = cuParamSetv(function, 0, &parameters.A, sizeof(parameters.A));
	if (result != CUDA_SUCCESS) {
		report("cuParamSetv() failed: " << result);
		return 1;
	}
	
	result = cuFuncSetBlockShape(function, 1, 1, 1);
	if (result != CUDA_SUCCESS) {
		report("cuFuncSetBlockShape() failed: " << result);
		return 1;
	}
	result = cuLaunchGrid(function, 1, 1);
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
	for (int i = 0; i < 9; i++) {
		if (i < 3 && !A_cpu[i] || i >= 3 && A_cpu[i]) {
			++errors;	
			std::cout << "%p" << i << " - " << A_cpu[i] << "\n";
		}
	}
	
	delete [] A_cpu;
	
	std::cout << "Pass/Fail : " << (!errors ? "Pass" : "Fail") << std::endl;
	
	return 0;
}

