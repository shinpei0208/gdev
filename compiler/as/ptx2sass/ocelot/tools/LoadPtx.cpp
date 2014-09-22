/*!
	\file genericmemory.cpp
	\brief CUDA Driver API version of the sequence program
*/

#include <ocelot/cuda/interface/CudaDriver.h>
#include <iostream>

#define report(x) std::cout << x << std::endl

std::ostream & operator<<(std::ostream &out, CUresult r) {
	out << "CUDA ERROR: " << (int)r;
	return out;
}

typedef cuda::CudaDriver driver;

int main(int argc, char **arg) {
	
	CUresult result = driver::cuInit(0);
	if (result != CUDA_SUCCESS) {
		report("cuInit() failed: " << result);
		return 1;
	}
	
	int driverVersion = 0;
	result = driver::cuDriverGetVersion(&driverVersion);
	if (result != CUDA_SUCCESS) {
		report("cuDriverGetVersion() failed: " << result);
	}
	
	int count = 0;
	result = driver::cuDeviceGetCount(&count);
	if (result != CUDA_SUCCESS) {
		report("cuDeviceGetCount() failed: " << result);
		return 1;
	}
	
	CUdevice device;
	result = driver::cuDeviceGet(&device, 0);
	if (result != CUDA_SUCCESS) {
		report("cuDeviceGet() failed: " << result);
		return 1;
	}
	
	char devName[256] = {0};
	result = driver::cuDeviceGetName(devName, 255, device);
	if (result != CUDA_SUCCESS) {
		report("cuDeviceGetName() failed: " << result);
		return 1;
	}
	
	if (argc < 2 || (argc == 2 && std::string(arg[1]) == "-h")) {
		std::cout << "usage: LoadPtx [ptx file] [type] [identifier]\n\n";
		std::cout << "  Loads a given PTX file. If an object is specified, attempts to obtain a handle\n\n";
		std::cout << "    type:\n";
		std::cout << "       -t - texture\n";
		std::cout << "       -k - kernel\n";
		std::cout << "       -g - global\n";
		std::cout << "\n\n" << count << " CUDA devices available. Using " << devName << std::endl;
		return 0;
	}
	
	int major, minor;
	result = driver::cuDeviceComputeCapability(&major, &minor, device);
	if (result != CUDA_SUCCESS) {
		report("cuDeviceComputeCapability() failed: " << result);
		return 1;
	}
	
	CUcontext ctx;
	CUmodule module;

	result = driver::cuCtxCreate(&ctx, 0, device);
	if (result != CUDA_SUCCESS) {
		report("cuCtxCreate() failed: " << result);
		return 1;
	}
	
	int pi = 0;
	result = driver::cuDeviceGetAttribute(&pi, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, device);
	if (result != CUDA_SUCCESS) {
		report("cuDeviceGetAttribute() failed: " << result);
	}
	
	result = driver::cuModuleLoad(&module, arg[1]);
	if (result != CUDA_SUCCESS) {
		report("cuModuleLoad() failed: " << result << " on module '" << arg[1] << "'");
		return 1;
	}
	
	std::cout << "Loaded module: " << arg[1] << std::endl;
	
	for (int i = 2; i < argc; i+=2) {
		if (i + 1 < argc) {
			std::string type = arg[i];
			if (type == "-g") {
				CUdeviceptr object;
				size_t bytes = 0;
				result = driver::cuModuleGetGlobal(&object, &bytes,module, arg[i+1]);
				if (result != CUDA_SUCCESS) {
					report("cuModuleGetGlobal() failed to obtain handle to global '" << arg[i+1]
						<< "' with result : " << result);
					return 1;
				}
				std::cout << "obtained handle to global '" << arg[i+1] << "' at address " 
					<< (void *)object << " of size " << bytes << " bytes" << std::endl;
			}
			else if (type == "-k") {
				CUfunction object;
				result = driver::cuModuleGetFunction(&object, module, arg[i+1]);
				if (result != CUDA_SUCCESS) {
					report("cuModuleGetFunction() failed to obtain handle to function " << arg[i+1]
						<< " with result: " << result);
					return 1;
				}
				std::cout << "obtained handle to kernel '" << arg[i+1] << "'" << std::endl;
			}
			else if (type == "-t") {
				CUtexref object;
				result = driver::cuModuleGetTexRef(&object, module, arg[i+1]);
				if (result != CUDA_SUCCESS) {
					report("cuModuleGetTexRef() failed to obtain handle to texture '" << 
						arg[i+1] << "' with result : " << result);
					return 1;
				}
				std::cout << "obtained handle to texture '" << arg[i+1] << "'" << std::endl;
			}
			else {
				std::cout << "Unknown definition type '" << type << "'" << std::endl;
				break;
			}
		}
		else {
			break;
		}
	}
	
	return 0;
}

