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

int cuda_test_float(float a, float b, char *path)
{
	int i, j, idx;
	CUresult res;
	CUdevice dev;
	CUcontext ctx;
	CUfunction function;
	CUmodule module;
	CUdeviceptr c_dev;
	float c;
	char fname[256];

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

	sprintf(fname, "%s/float_gpu.cubin", path);
	res = cuModuleLoad(&module, fname);
	if (res != CUDA_SUCCESS) {
		printf("cuModuleLoad() failed\n");
		return -1;
	}
	res = cuModuleGetFunction(&function, module, "add");
	if (res != CUDA_SUCCESS) {
		printf("cuModuleGetFunction() failed\n");
		return -1;
	}
	res = cuFuncSetSharedSize(function, 0);
	if (res != CUDA_SUCCESS) {
		printf("cuFuncSetSharedSize() failed\n");
		return -1;
	}
	res = cuFuncSetBlockShape(function, 1, 1, 1);
	if (res != CUDA_SUCCESS) {
		printf("cuFuncSetBlockShape() failed\n");
		return -1;
	}

	/* c */
	res = cuMemAlloc(&c_dev, sizeof(float));
	if (res != CUDA_SUCCESS) {
		printf("cuMemAlloc (c) failed\n");
		return -1;
	}

	/* set kernel parameters */
	res = cuParamSetf(function, 0, a);	
	if (res != CUDA_SUCCESS) {
		printf("cuParamSeti (a) failed: res = %lu\n", (unsigned long)res);
		return -1;
	}
	res = cuParamSetf(function, 4, b);
	if (res != CUDA_SUCCESS) {
		printf("cuParamSeti (b) failed: res = %lu\n", (unsigned long)res);
		return -1;
	}
	res = cuParamSetv(function, 8, &c_dev, 8);
	if (res != CUDA_SUCCESS) {
		printf("cuParamSeti (c_dev) failed: res = %lu\n", (unsigned long)res);
		return -1;
	}
	res = cuParamSetSize(function, 16);
	if (res != CUDA_SUCCESS) {
		printf("cuParamSetSize failed: res = %lu\n", (unsigned long)res);
		return -1;
	}

	/* launch the kernel */
	res = cuLaunchGrid(function, 1, 1);
	if (res != CUDA_SUCCESS) {
		printf("cuLaunchGrid failed: res = %lu\n", (unsigned long)res);
		return -1;
	}
	cuCtxSynchronize();

	/* download c */
	res = cuMemcpyDtoH(&c, c_dev, sizeof(float));
	if (res != CUDA_SUCCESS) {
		printf("cuMemcpyDtoH (c) failed: res = %lu\n", (unsigned long)res);
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

	/* check the results */
	if (c != a + b) {
		printf("c = %f\n", c);
		printf("a + b = %f\n", a + b);
		return -1;
	}

	return 0;
}


