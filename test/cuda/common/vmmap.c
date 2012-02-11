#include <cuda.h>
#ifdef __KERNEL__ /* just for measurement */
#include <linux/vmalloc.h>
#include <linux/time.h>
#define printf printk
#define malloc vmalloc
#define free vfree
#define gettimeofday(x, y) do_gettimeofday(x)
#else /* just for measurement */
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#endif

/* tvsub: ret = x - y. */
static inline void tvsub(struct timeval *x, 
						 struct timeval *y, 
						 struct timeval *ret)
{
	ret->tv_sec = x->tv_sec - y->tv_sec;
	ret->tv_usec = x->tv_usec - y->tv_usec;
	if (ret->tv_usec < 0) {
		ret->tv_sec--;
		ret->tv_usec += 1000000;
	}
}

int cuda_test_madd(unsigned int n, char *path)
{
	int i, j, idx;
	CUresult res;
	CUdevice dev;
	CUcontext ctx;
	CUfunction function;
	CUmodule module;
	CUdeviceptr a_dev, b_dev, c_dev;
	unsigned int *a_buf, *b_buf, *c_buf;
	unsigned long long int a_phys, b_phys, c_phys;
	int block_x, block_y, grid_x, grid_y;
	char fname[256];
	int ret = 0;

	/* block_x * block_y should not exceed 512. */
	block_x = n < 16 ? n : 16;
	block_y = n < 16 ? n : 16;
	grid_x = n / block_x;
	if (n % block_x != 0)
		grid_x++;
	grid_y = n / block_y;
	if (n % block_y != 0)
		grid_y++;

	res = cuInit(0);
	if (res != CUDA_SUCCESS) {
		printf("cuInit failed: res = %lu\n", (unsigned long)res);
		return -1;
	}

	res = cuDeviceGet(&dev, 0);
	if (res != CUDA_SUCCESS) {
		printf("cuDeviceGet failed: res = %lu\n", (unsigned long)res);
		return -1;
	}

	res = cuCtxCreate(&ctx, 0, dev);
	if (res != CUDA_SUCCESS) {
		printf("cuCtxCreate failed: res = %lu\n", (unsigned long)res);
		return -1;
	}

	sprintf(fname, "%s/madd_gpu.cubin", path);
	res = cuModuleLoad(&module, fname);
	if (res != CUDA_SUCCESS) {
		printf("cuModuleLoad() failed\n");
		return -1;
	}
	res = cuModuleGetFunction(&function, module, "_Z3addPjS_S_j");
	if (res != CUDA_SUCCESS) {
		printf("cuModuleGetFunction() failed\n");
		return -1;
	}
	res = cuFuncSetBlockShape(function, block_x, block_y, 1);
	if (res != CUDA_SUCCESS) {
		printf("cuFuncSetBlockShape() failed\n");
		return -1;
	}

	/* a[] */
	res = cuMemAlloc(&a_dev, n*n * sizeof(unsigned int));
	if (res != CUDA_SUCCESS) {
		printf("cuMemAlloc (a) failed\n");
		return -1;
	}
	res = cuMemMap((void**)&a_buf, a_dev, n*n * sizeof(unsigned int));
	if (res != CUDA_SUCCESS) {
		printf("cuMemMap (a) failed\n");
		return -1;
	}
	res = cuMemGetPhysAddr(&a_phys, (void*)a_buf);
	if (res != CUDA_SUCCESS) {
		printf("cuMemGetPhysAddress (a) failed\n");
		return -1;
	}
	printf("a[]: Physical Address 0x%llx\n", a_phys);

	/* b[] */
	res = cuMemAlloc(&b_dev, n*n * sizeof(unsigned int));
	if (res != CUDA_SUCCESS) {
		printf("cuMemAlloc (b) failed\n");
		return -1;
	}
	res = cuMemMap((void**)&b_buf, b_dev, n*n * sizeof(unsigned int));
	if (res != CUDA_SUCCESS) {
		printf("cuMemMap (b) failed\n");
		return -1;
	}
	res = cuMemGetPhysAddr(&b_phys, (void*)b_buf);
	if (res != CUDA_SUCCESS) {
		printf("cuMemGetPhysAddress (b) failed\n");
		return -1;
	}
	printf("b[]: Physical Address 0x%llx\n", b_phys);

	/* c[] */
	res = cuMemAlloc(&c_dev, n*n * sizeof(unsigned int));
	if (res != CUDA_SUCCESS) {
		printf("cuMemAlloc (c) failed\n");
		return -1;
	}
	res = cuMemMap((void**)&c_buf, c_dev, n*n * sizeof(unsigned int));
	if (res != CUDA_SUCCESS) {
		printf("cuMemMap (c) failed\n");
		return -1;
	}
	res = cuMemGetPhysAddr(&c_phys, (void*)c_buf);
	if (res != CUDA_SUCCESS) {
		printf("cuMemGetPhysAddress (c) failed\n");
		return -1;
	}
	printf("c[]: Physical Address 0x%llx\n", c_phys);

#if 1
	/* initialize A[] & B[] */
	for (i = 0; i < n; i++) {
		for(j = 0; j < n; j++) {
			idx = i * n + j;
			a_buf[idx] = i;
			b_buf[idx] = i + 1;
		}
	}
#endif

	/* set kernel parameters */
	res = cuParamSeti(function, 0, a_dev);	
	if (res != CUDA_SUCCESS) {
		printf("cuParamSeti (a) failed: res = %lu\n", (unsigned long)res);
		return -1;
	}
	res = cuParamSeti(function, 4, a_dev >> 32);
	if (res != CUDA_SUCCESS) {
		printf("cuParamSeti (a) failed: res = %lu\n", (unsigned long)res);
		return -1;
	}
	res = cuParamSeti(function, 8, b_dev);
	if (res != CUDA_SUCCESS) {
		printf("cuParamSeti (b) failed: res = %lu\n", (unsigned long)res);
		return -1;
	}
	res = cuParamSeti(function, 12, b_dev >> 32);
	if (res != CUDA_SUCCESS) {
		printf("cuParamSeti (b) failed: res = %lu\n", (unsigned long)res);
		return -1;
	}
	res = cuParamSeti(function, 16, c_dev);
	if (res != CUDA_SUCCESS) {
		printf("cuParamSeti (c) failed: res = %lu\n", (unsigned long)res);
		return -1;
	}
	res = cuParamSeti(function, 20, c_dev >> 32);
	if (res != CUDA_SUCCESS) {
		printf("cuParamSeti (c) failed: res = %lu\n", (unsigned long)res);
		return -1;
	}
	res = cuParamSeti(function, 24, n);
	if (res != CUDA_SUCCESS) {
		printf("cuParamSeti (c) failed: res = %lu\n", (unsigned long)res);
		return -1;
	}
	res = cuParamSetSize(function, 28);
	if (res != CUDA_SUCCESS) {
		printf("cuParamSetSize failed: res = %lu\n", (unsigned long)res);
		return -1;
	}

	/* launch the kernel */
	res = cuLaunchGrid(function, grid_x, grid_y);
	if (res != CUDA_SUCCESS) {
		printf("cuLaunchGrid failed: res = %lu\n", (unsigned long)res);
		return -1;
	}
	cuCtxSynchronize();

#if 0
	/* check the results */
	i = j = idx = 0;
	while (i < n) {
		while (j < n) {
			idx = i * n + j;
			if (c_buf[idx] != a_buf[idx] + b_buf[idx]) {
				printf("c[%d] = %d\n", idx, c_buf[idx]);
				printf("a[%d]+b[%d] = %d\n", idx, idx, a_buf[idx]+b_buf[idx]);
				ret = -1;
			}
			j++;
			if (ret)
				break;
		}
		i++;
		if (ret)
			break;
	}
#endif

	res = cuMemUnmap((void*)a_buf);
	if (res != CUDA_SUCCESS) {
		printf("cuMemUnmap (a) failed: res = %lu\n", (unsigned long)res);
		return -1;
	}
	res = cuMemFree(a_dev);
	if (res != CUDA_SUCCESS) {
		printf("cuMemFree (a) failed: res = %lu\n", (unsigned long)res);
		return -1;
	}
	res = cuMemUnmap((void*)b_buf);
	if (res != CUDA_SUCCESS) {
		printf("cuMemUnmap (b) failed: res = %lu\n", (unsigned long)res);
		return -1;
	}
	res = cuMemFree(b_dev);
	if (res != CUDA_SUCCESS) {
		printf("cuMemFree (b) failed: res = %lu\n", (unsigned long)res);
		return -1;
	}
	res = cuMemUnmap((void*)c_buf);
	if (res != CUDA_SUCCESS) {
		printf("cuMemUnmap (c) failed: res = %lu\n", (unsigned long)res);
		return -1;
	}
	res = cuMemFree(c_dev);
	if (res != CUDA_SUCCESS) {
		printf("cuMemFree (c) failed: res = %lu\n", (unsigned long)res);
		return -1;
	}

	res = cuModuleUnload(module);
	if (res != CUDA_SUCCESS) {
		printf("cuModuleUnload failed: res = %lu\n", (unsigned long)res);
		return -1;
	}

	res = cuCtxDestroy(ctx);
	if (res != CUDA_SUCCESS) {
		printf("cuCtxDestroy failed: res = %lu\n", (unsigned long)res);
		return -1;
	}

	return ret;
}


