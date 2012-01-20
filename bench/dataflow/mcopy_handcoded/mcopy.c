/*
 * mcopy.c -- User code for matrix copy benchmark
 *
 * Michael McThrow
 */

#include "mcopy.h"

int mcopy_gpu_init(struct device_info *device_info)
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

	return 0;
}

int mcopy_gpu_close(struct device_info *device_info)
{
	CUresult res;

	/* printf("madd_gpu_close called.\n"); */

        res = cuCtxDestroy(device_info->context);
        if (res != CUDA_SUCCESS) {
                printf("cuCtxDestroy failed: res = %lu\n", (unsigned long)res);
                return -1;
        }

	return 0;
}

int mcopy_gpu(struct device_info *device_info, CUdeviceptr *a_dev, CUdeviceptr
 *b_dev, CUdeviceptr *c_dev, unsigned int rows, unsigned int cols)
{
	CUresult res;

	/* copy matrix a_dev to c_dev */
	if ((res = cuMemcpyDtoD(*c_dev, *a_dev, rows * cols *
	 sizeof(unsigned int))) != CUDA_SUCCESS) {
		printf("cuMemcpyDtoD failed: res = %lu\n", (unsigned long)res);
		return -1;
	}

	return 0;
}
