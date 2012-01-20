/*
 * mmult.c -- User code for matrix multiplication benchmark
 *
 * Michael McThrow
 */

#include "mmult.h"

static int setParams(CUfunction kernel, CUdeviceptr a_dev, CUdeviceptr b_dev,
 CUdeviceptr c_dev, unsigned int rows, unsigned int cols);

int mmult_gpu_init(struct device_info *device_info)
{
	char fname[256];
	CUresult res;

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

static int setParams(CUfunction kernel, CUdeviceptr a_dev, CUdeviceptr b_dev,
 CUdeviceptr c_dev, unsigned int rows, unsigned int cols)
{
	CUresult res;
	int offset = 0;

	/* printf("setParams called.\n"); */

	/* a_dev */
	if ((res = cuParamSetv(kernel, offset, a_dev, sizeof(a_dev))) !=
	 CUDA_SUCCESS) {
                printf("cuParamSetv (a) failed: res = %lu\n",
		 (unsigned long)res);
                return -1;
	}

	offset += sizeof(a_dev);

	/* b_dev */
	if ((res = cuParamSetv(kernel, offset, b_dev, sizeof(b_dev))) !=
	 CUDA_SUCCESS) {
                printf("cuParamSetv (b) failed: res = %lu\n",
		 (unsigned long)res);
                return -1;
	}

	offset += sizeof(b_dev);

	/* c_dev */
	if ((res = cuParamSetv(kernel, offset, c_dev, sizeof(c_dev))) !=
	 CUDA_SUCCESS) {
                printf("cuParamSetv (c) failed: res = %lu\n",
		 (unsigned long)res);
                return -1;
	}

	offset += sizeof(c_dev);

	/* rows */
	if ((res = cuParamSeti(kernel, offset, rows)) != CUDA_SUCCESS) {
		printf("cuParamSeti (rows) failed: res = %lu\n",
		 (unsigned long)res);
		return -1;
	}

	offset += sizeof(rows);

	/* cols */
	if ((res = cuParamSeti(kernel, offset, cols)) != CUDA_SUCCESS) {
		printf("cuParamSeti (cols) failed: res = %lu\n",
		 (unsigned long)res);
		return -1;
	}

	offset += sizeof(cols);

	/* set size of the parameters */
	if ((res = cuParamSetSize(kernel, offset)) != CUDA_SUCCESS) {
                printf("cuParamSetSize failed: res = %lu\n",
		 (unsigned long)res);
                return -1;
	}

	return 0;
}

int mmult_gpu_close(struct device_info *device_info)
{
	CUresult res;

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

int mmult_gpu(struct device_info *device_info, unsigned int *a, unsigned int *b,
 unsigned int *c, unsigned int rows, unsigned int cols)
{
	CUdeviceptr a_dev, b_dev, c_dev;	/* device mem. copies of */
	CUresult res;

	/* initialize gpu */
	if (mmult_gpu_init(device_info))
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

	/* set kernel parameters */
	void *kernel_params[] = {&a_dev, &b_dev, &c_dev, &rows, &cols};

	/* execute kernel */
	unsigned int gridWidth = cols >> X_THREADS_PER_BLOCK_SHIFT;
	unsigned int gridHeight = rows >> Y_THREADS_PER_BLOCK_SHIFT;

	unsigned int shmemBytes = 0x40; /* random value */

	/*
	if ((res = cuFuncSetSharedSize(device_info->kernel, shmemBytes)) !=
	 CUDA_SUCCESS) {
		printf("cuFuncSetSharedSize() failed\n");
		return -1;
	}

	if ((res = cuFuncSetBlockShape(device_info->kernel, X_THREADS_PER_BLOCK,
	 Y_THREADS_PER_BLOCK, 1)) != CUDA_SUCCESS) {
                printf("cuFuncSetBlockShape() failed\n");
                return -1;
	}

	if (setParams(device_info->kernel, a_dev, b_dev, c_dev, rows, cols))
		return -1;

	printf("Launch grid.\n");

	if ((res = cuLaunchGrid(device_info->kernel, gridWidth, gridHeight)) !=
	 CUDA_SUCCESS) {
                printf("cuLaunchGrid failed: res = %lu\n", (unsigned long)res);
                return -1;
	}

	printf("Grid was successfully launched.\n");
	*/

	/* printf("Launch kernel.\n"); */

	if ((res = cuLaunchKernel(device_info->kernel, gridWidth, gridHeight, 1,
	 X_THREADS_PER_BLOCK, Y_THREADS_PER_BLOCK, 1, shmemBytes, 0,
	 kernel_params, 0)) != CUDA_SUCCESS) {
		printf("cuLaunchKernel failed: res = %lu\n",
		 (unsigned long)res);
		return -1;
	}

	/* printf("Kernel was successfully launched.\n"); */

	/* copy result from device memory to host memory */
	if ((res = cuMemcpyDtoH(c, c_dev, rows * cols * sizeof(unsigned int)))
	 != CUDA_SUCCESS) {
		printf("cuMemcpyDtoH (c) failed: res = %lu\n",
		 (unsigned long)res);
		return -1;
	}

	/* printf("Result obtained.\n"); */

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
	if (mmult_gpu_close(device_info))
		return -1;

	return 0;
}
