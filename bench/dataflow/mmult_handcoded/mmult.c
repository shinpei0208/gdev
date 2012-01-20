/*
 * mmult.c -- User code for matrix multiplication benchmark
 *
 * Michael McThrow
 */

#include "mmult.h"

int mmult_gpu_init(struct device_info *device_info)
{
	char fname[256];
	CUresult res;

	/* printf("madd_gpu_init called.\n"); */

	/* Initialization */
	if ((res = cuInit(0)) != CUDA_SUCCESS) {
		printf("cuInit failed: res = %lu\n", (unsigned long)res);
		return -1;
	}

	if ((res = cuDeviceGet(&device_info->dev, 0)) != CUDA_SUCCESS) {
		printf("cuDeviceGet failed: res = %lu\n", (unsigned long)res);
		return -1;
	}

	if ((res = cuCtxCreate(&device_info->context, 0, device_info->dev)) !=
	 CUDA_SUCCESS) {
		printf("cuCtxCreate failed: res = %lu\n", (unsigned long)res);
		return -1;
	}

	/* binary files are located in the same directory as the source code */
	if ((res = cuModuleLoad(&device_info->module, MODULE_FILE_NAME)) !=
	 CUDA_SUCCESS) {
		printf("cuModuleLoad() failed\n");
		return -1;
	}

	if ((res = cuModuleGetFunction(&device_info->kernel, 
	 device_info->module, KERNEL_NAME)) != CUDA_SUCCESS) {
		printf("cuModuleGetFunction() failed\n");
		return -1;
	}

	return 0;
}

int mmult_gpu_close(struct device_info *device_info)
{
	CUresult res;

	/* printf("mmult_gpu_close called.\n"); */

        res = cuModuleUnload(device_info->module);
        if (res != CUDA_SUCCESS) {
                printf("cuModuleUnload failed: res = %lu\n", (unsigned long)res);
                return -1;
        }

        res = cuCtxDestroy(device_info->context);
        if (res != CUDA_SUCCESS) {
                printf("cuCtxDestroy failed: res = %lu\n", (unsigned long)res);
                return -1;
        }

	return 0;
}

int mmult_gpu(struct device_info *device_info, CUdeviceptr *a_dev, CUdeviceptr
 *b_dev, CUdeviceptr *c_dev, unsigned int rows, unsigned int cols)
{
	CUresult res;

	/* set kernel parameters */
	void *kernel_params[] = {a_dev, b_dev, c_dev, &rows, &cols};

	/* execute kernel */
	unsigned int gridWidth = cols >> X_THREADS_PER_BLOCK_SHIFT;
	unsigned int gridHeight = rows >> Y_THREADS_PER_BLOCK_SHIFT;

	unsigned int shmemBytes = 0x40; /* random value */

	if ((res = cuLaunchKernel(device_info->kernel, gridWidth, gridHeight, 1,
	 X_THREADS_PER_BLOCK, Y_THREADS_PER_BLOCK, 1, shmemBytes, 0,
	 kernel_params, 0)) != CUDA_SUCCESS) {
		printf("cuLaunchKernel failed: res = %lu\n",
		 (unsigned long)res);
		return -1;
	}

	return 0;
}
