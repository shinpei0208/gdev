/*!
	\file genericmemory.cpp
	\brief CUDA Driver API version of the sequence program
*/

#include <ocelot/cuda/interface/cuda_runtime.h>
#include <ocelot/api/interface/ocelot.h>
#include <iostream>
#include <fstream>

#define report(x) std::cout << x << std::endl

int main() {
	const int N = 9;
	
	int bytes = sizeof(int) * N;
	int *A_cpu = 0;
	int *A_gpu = 0;
	
	A_cpu = new int[N];
	for (int i = 0; i < N; i++) {
		A_cpu[i] = -1;
	}
	
	cudaError_t result;
	
	result = cudaMalloc((void **)&A_gpu, bytes);
	if (result != cudaSuccess) {
		report("cudaMalloc() failed");
		return 1;
	}
	
	result = cudaMemcpy(A_gpu, A_cpu, bytes, cudaMemcpyHostToDevice);
	if (result != cudaSuccess) {
		report("cudaMemcpy host-to-device failed");
		return 2;
	}
	
	// load module
	std::ifstream file("ocelot/cuda/test/memory/generic.ptx");
	ocelot::registerPTXModule(file, "generic.ptx");
	
	// configure call
	result = cudaConfigureCall(dim3(1,1,1), dim3(1,1,1), 0, 0);
	if (result != cudaSuccess) {
		report("cudaConfigureCall() - failed");
		return 3;
	}
	
	result = cudaSetupArgument(&A_gpu, sizeof(int *), 0);
	if (result != cudaSuccess) {
		report("cudaSetupArgument() - failed");
		return 4;
	}

	ocelot::launch("generic.ptx", "genericmemory");
	
	result = cudaMemcpy(A_cpu, A_gpu, bytes, cudaMemcpyDeviceToHost);
	if (result != cudaSuccess) {
		report("cudaMemcpy() - device-to-host failed");
		return 4;
	}

	int errors = 0;
	for (int i = 0; i < 9; i++) {
		if ((i < 3 && !A_cpu[i]) || (i >= 3 && A_cpu[i])) {
			++errors;	
			std::cout << "%p" << i << " - " << A_cpu[i] << "\n";
		}
	}
	
	delete [] A_cpu;
	cudaFree(A_gpu);
	
	std::cout << "Pass/Fail : " << (!errors ? "Pass" : "Fail") << std::endl;
	
	return 0;
}

