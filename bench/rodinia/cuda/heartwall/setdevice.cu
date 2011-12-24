////////////////////////////////////////////////////////////////////////////////
// Set Device
////////////////////////////////////////////////////////////////////////////////

void setdevice(void){

	// variables
	int num_devices;
	int device;

	cudaGetDeviceCount(&num_devices);
	if (num_devices > 1) {
		
		// variables
		int max_multiprocessors; 
		int max_device;
		cudaDeviceProp properties;

		// initialize variables
		max_multiprocessors = 0;
		max_device = 0;
		
		for (device = 0; device < num_devices; device++) {
			cudaGetDeviceProperties(&properties, device);
			if (max_multiprocessors < properties.multiProcessorCount) {
				max_multiprocessors = properties.multiProcessorCount;
				max_device = device;
			}
		}
		cudaSetDevice(max_device);
	}

}
