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

int cuda_test_loop_repeated(unsigned int n, int sec, int id, char *path)
{
	int i, j, idx;
	CUresult res;
	CUdevice dev;
	CUcontext ctx;
	CUfunction function;
	CUmodule module;
	CUdeviceptr d_data;
	unsigned int *data = (unsigned int *) malloc (n * sizeof(unsigned int));
	int block_x, block_y, grid_x, grid_y;
	char fname[256];
	struct timeval tv_start, tv_now, tv;

	block_x = 1;
	block_y = 1;
	grid_x = 1;
	grid_y = 1;

	gettimeofday(&tv_start, NULL);

	res = cuInit(0);
	if (res != CUDA_SUCCESS) {
		printf("cuInit failed: res = %lu\n", (unsigned long)res);
		return -1;
	}

	res = cuDeviceGet(&dev, id);
	if (res != CUDA_SUCCESS) {
		printf("cuDeviceGet failed: res = %lu\n", (unsigned long)res);
		return -1;
	}

	res = cuCtxCreate(&ctx, 0, dev);
	if (res != CUDA_SUCCESS) {
		printf("cuCtxCreate failed: res = %lu\n", (unsigned long)res);
		return -1;
	}

	sprintf(fname, "%s/loop_gpu.cubin", path);
	res = cuModuleLoad(&module, fname);
	if (res != CUDA_SUCCESS) {
		printf("cuModuleLoad() failed\n");
		return -1;
	}
	res = cuModuleGetFunction(&function, module, "_Z4loopPjjj");
	if (res != CUDA_SUCCESS) {
		printf("cuModuleGetFunction() failed\n");
		return -1;
	}
	res = cuFuncSetSharedSize(function, 0); 
	if (res != CUDA_SUCCESS) {
		printf("cuFuncSetSharedSize() failed\n");
		return -1;
	}
	res = cuFuncSetBlockShape(function, block_x, block_y, 1);
	if (res != CUDA_SUCCESS) {
		printf("cuFuncSetBlockShape() failed\n");
		return -1;
	}

	res = cuMemAlloc(&d_data, n * sizeof(unsigned int));
	if (res != CUDA_SUCCESS) {
		printf("cuMemAlloc failed\n");
		return -1;
	}

repeat:
	usleep(100);
	res = cuMemcpyHtoD(d_data, data, n * sizeof(unsigned int));
	if (res != CUDA_SUCCESS) {
		printf("cuMemcpyHtoD failed: res = %lu\n", (unsigned long)res);
		return -1;
	}

	static int counter = 0;
//repeat:
//	usleep(100);
//	counter++;
//	usleep(10 * (rand() % 10));

	/* set kernel parameters */
	res = cuParamSeti(function, 0, d_data);	
	if (res != CUDA_SUCCESS) {
		printf("cuParamSeti failed: res = %lu\n", (unsigned long)res);
		return -1;
	}
	res = cuParamSeti(function, 4, d_data >> 32);
	if (res != CUDA_SUCCESS) {
		printf("cuParamSeti failed: res = %lu\n", (unsigned long)res);
		return -1;
	}
	res = cuParamSeti(function, 8, n * sizeof(unsigned int));
	if (res != CUDA_SUCCESS) {
		printf("cuParamSeti failed: res = %lu\n", (unsigned long)res);
		return -1;
	}
	res = cuParamSeti(function, 12, (n - (rand() % 1000)));
	if (res != CUDA_SUCCESS) {
		printf("cuParamSeti failed: res = %lu\n", (unsigned long)res);
		return -1;
	}
	res = cuParamSetSize(function, 16);
	if (res != CUDA_SUCCESS) {
		printf("cuParamSetSize failed: res = %lu\n", (unsigned long)res);
		return -1;
	}
	
	res = cuLaunchGrid(function, grid_x, grid_y);
	if (res != CUDA_SUCCESS) {
		printf("cuLaunchGrid failed: res = %lu\n", (unsigned long)res);
		return -1;
	}
	cuCtxSynchronize();

//	gettimeofday(&tv_now, NULL);
//	tvsub(&tv_now, &tv_start, &tv);
//	printf("%lu:%lu\n", tv.tv_sec, tv.tv_usec);
//	if (tv.tv_sec < sec) {
//		goto repeat;
//	}

	res = cuMemcpyDtoH(data, d_data, n * sizeof(unsigned int));
	if (res != CUDA_SUCCESS) {
		printf("cuMemcpyDtoH failed: res = %lu\n", (unsigned long)res);
		return -1;
	}

	gettimeofday(&tv_now, NULL);
	tvsub(&tv_now, &tv_start, &tv);
	printf("%lu:%lu\n", tv.tv_sec, tv.tv_usec);
	if (tv.tv_sec < sec) {
		goto repeat;
	}

	res = cuMemFree(d_data);
	if (res != CUDA_SUCCESS) {
		printf("cuMemFree failed: res = %lu\n", (unsigned long)res);
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

	free(data);

	return 0;
}
