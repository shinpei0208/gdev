/*
 * madd.c -- User code for matrix addition benchmark
 *
 * Michael McThrow
 */

#include "madd.h"

/* declaration of global key counter */
unsigned int key_counter = 0;

int madd_gpu_init(struct device_info *device_info)
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

int madd_gpu_close(struct device_info *device_info)
{
	CUresult res;

	/* printf("madd_gpu_close called.\n"); */

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

int madd_gpu(int a_key, int b_key, int *c_key, unsigned int rows, unsigned int
 cols)
{
	CUdeviceptr a_dev, b_dev, c_dev;	/* pointers to shmem */
	int a_shmid, b_shmid, c_shmid;
	CUresult res;
	struct device_info device_info;
	int matrix_alloc_size = rows * cols * sizeof(unsigned int);

	/* printf("madd_gpu called.\n"); */

	/* initialize gpu */
	if (madd_gpu_init(&device_info))
		return -1;

	/* get shmids of a and b */
	if ((res = cuShmGet(&a_shmid, a_key, matrix_alloc_size, 0)) !=
	 CUDA_SUCCESS) {
                printf("cuShmGet (a) failed\n");
                return -1;
	}

	if ((res = cuShmGet(&b_shmid, b_key, matrix_alloc_size, 0)) !=
	 CUDA_SUCCESS) {
                printf("cuShmGet (b) failed\n");
                return -1;
	}

	/* allocate shared memory for matrix c */
	*c_key = key_counter++;

	if ((res = cuShmGet(&c_shmid, *c_key, matrix_alloc_size, 0)) !=
	 CUDA_SUCCESS) {
                printf("cuShmGet (c) failed\n");
                return -1;
	}

	/* attach a, b, and c to current context's address space */
	if ((res = cuShmAt(&a_dev, a_shmid, 0)) != CUDA_SUCCESS) {
		printf("cuShmAt (a) failed\n");
		return -1;
	}

	if ((res = cuShmAt(&b_dev, b_shmid, 0)) != CUDA_SUCCESS) {
		printf("cuShmAt (b) failed\n");
		return -1;
	}

	if ((res = cuShmAt(&c_dev, c_shmid, 0)) != CUDA_SUCCESS) {
		printf("cuShmAt (c) failed\n");
		return -1;
	}

	/* set kernel parameters */
	void *kernel_params[] = {&a_dev, &b_dev, &c_dev, &rows, &cols};

	/* execute kernel */
	unsigned int gridWidth = cols >> X_THREADS_PER_BLOCK_SHIFT;
	unsigned int gridHeight = rows >> Y_THREADS_PER_BLOCK_SHIFT;

	unsigned int shmemBytes = 0x40; /* random value */

	/* printf("Launch kernel.\n"); */

	if ((res = cuLaunchKernel(device_info.kernel, gridWidth, gridHeight,
	 1, X_THREADS_PER_BLOCK, Y_THREADS_PER_BLOCK, 1, shmemBytes, 0,
	 kernel_params, 0)) != CUDA_SUCCESS) {
		printf("cuLaunchKernel failed: res = %lu\n",
		 (unsigned long)res);
		return -1;
	}

	/* printf("Kernel was successfully launched.\n"); */

	/* printf("Result obtained.\n"); */

	/* detach a, b, and c */
	if ((res = cuShmDt(a_dev)) != CUDA_SUCCESS) {
		printf("cuShmDt (a) failed: res = %u\n", res);
		return -1;
	}

	if ((res = cuShmDt(b_dev)) != CUDA_SUCCESS) {
		printf("cuShmDt (b) failed: res = %u\n", res);
		return -1;
	}

	if ((res = cuShmDt(c_dev)) != CUDA_SUCCESS) {
		printf("cuShmDt (c) failed: res = %u\n", res);
		return -1;
	}

	/* close gpu */
	if (madd_gpu_close(&device_info))
		return -1;

	return 0;
}

int shmem_device_copy(int key, int size, unsigned int *matrix, int toDevice)
{
        int shmid;
        CUresult res;
        CUdeviceptr addr;

        if ((res = cuShmGet(&shmid, key, size, 0)) != CUDA_SUCCESS) {
                printf("cuShmGet failed: res = %u\n", res);

                return -1;
        }

        /* attach a local pointer to shared memory */
        if ((res = cuShmAt(&addr, shmid, 0)) != CUDA_SUCCESS) {
                printf("cuShmAt failed: res = %u\n", res);

                return -1;
        }

        /* copy current matrix from host memory to shared memory */
        if (toDevice) {
                if ((res = cuMemcpyHtoD(addr, matrix, size)) != CUDA_SUCCESS) {
                        printf("cuMemcpyHtoD failed: res = %u\n", res);

                        return -1;
                }
        }
        else {
                if ((res = cuMemcpyDtoH(matrix, addr, size)) != CUDA_SUCCESS) {
                        printf("cuMemcpyDtoH failed: res = %u\n", res);

                        return -1;
                }
        }

        /* detach local pointer */
        if ((res = cuShmDt(addr)) != CUDA_SUCCESS) {
                printf("cuShmDt failed: res = %u\n", res);

                return -1;
        }

        return 0;
}

int free_shmem(int key, int size)
{
        int shmid;
        CUresult res;

        if ((res = cuShmGet(&shmid, key, size, 0)) != CUDA_SUCCESS) {
                printf("cuShmGet failed: res = %u\n", res);
                return -1;
        }

	/*
        if ((res = cuShmCtl(shmid, GDEV_IPC_RMID, NULL)) != CUDA_SUCCESS) {
                printf("cuShmCtl failed: res = %u\n", res);
                return -1;
        }
	*/

        return 0;
}
