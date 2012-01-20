/*
 * mcopy.c -- User code for matrix addition benchmark
 *
 * Michael McThrow
 */

#include "mcopy.h"

static int setParams(CUfunction kernel, CUdeviceptr a_dev, CUdeviceptr b_dev,
 CUdeviceptr c_dev, unsigned int rows, unsigned int cols);

int mcopy_gpu_init(struct device_info *device_info)
{
	char fname[256];
	CUresult res;

	/* printf("mcopy_gpu_init called.\n"); */

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

	/* printf("mcopy_gpu_close called.\n"); */

        res = cuCtxDestroy(device_info->context);
        if (res != CUDA_SUCCESS) {
                printf("cuCtxDestroy failed: res = %lu\n", (unsigned long)res);
                return -1;
        }

	return 0;
}

int mcopy_gpu(struct device_info *device_info, unsigned int *a, unsigned int *b,
 unsigned int *c, unsigned int rows, unsigned int cols)
{
	CUdeviceptr a_dev, b_dev, c_dev;	/* device mem. copies of */
	CUresult res;

	/* initialize gpu */
	if (mcopy_gpu_init(device_info))
		return -1;

	/* allocate device memory for matrices a, b, and c */
	if (cuMemAlloc(&a_dev, rows * cols * sizeof(unsigned int)) !=
	 CUDA_SUCCESS) {
                printf("cuMemAlloc (a) failed\n");
                return -1;
	}

	if (cuMemAlloc(&b_dev, rows * cols * sizeof(unsigned int)) !=
	 CUDA_SUCCESS) {
                printf("cuMemAlloc (b) failed\n");
                return -1;
	}

	if (cuMemAlloc(&c_dev, rows * cols * sizeof(unsigned int)) !=
	 CUDA_SUCCESS) {
                printf("cuMemAlloc (c) failed\n");
                return -1;
	}

	/* copy a and b from host memory to device memory */
	if ((res = cuMemcpyHtoD(a_dev, a, rows * cols * sizeof(unsigned int)))
	 != CUDA_SUCCESS) {
                printf("cuMemcpyHtoD (a) failed: res = %lu\n",
		 (unsigned long)res);
                return -1;
	}

	if ((res = cuMemcpyHtoD(b_dev, b, rows * cols * sizeof(unsigned int)))
	 != CUDA_SUCCESS) {
                printf("cuMemcpyHtoD (b) failed: res = %lu\n",
		 (unsigned long)res);
                return -1;
	}

	/* copy a to c */
	if ((res = cuMemcpyDtoD(c_dev, a_dev, rows * cols * sizeof(unsigned int)))
	 != CUDA_SUCCESS) {
		printf("cuMemcpyDtoD (a) failed: res = %lu\n",
		 (unsigned long)res);
		return -1;
	}

	/* copy result from device memory to host memory */
	if ((res = cuMemcpyDtoH(c, c_dev, rows * cols * sizeof(unsigned int)))
	 != CUDA_SUCCESS) {
		printf("cuMemcpyDtoH (c) failed: res = %lu\n",
		 (unsigned long)res);
		return -1;
	}

	/* cleanup */
        if ((res = cuMemFree(a_dev)) != CUDA_SUCCESS) {
                printf("cuMemFree (a) failed: res = %lu\n", (unsigned long)res);
                return -1;
        }

        if ((res = cuMemFree(b_dev)) != CUDA_SUCCESS) {
                printf("cuMemFree (b) failed: res = %lu\n", (unsigned long)res);
                return -1;
        }

        if ((res = cuMemFree(c_dev)) != CUDA_SUCCESS) {
                printf("cuMemFree (c) failed: res = %lu\n", (unsigned long)res);
                return -1;
        }

	/* close gpu */
	if (mcopy_gpu_close(device_info))
		return -1;

	return 0;
}
