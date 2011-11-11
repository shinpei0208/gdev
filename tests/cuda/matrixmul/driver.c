#define PTX

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <sys/time.h>

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

int test_matrixmul(unsigned int n)
{
	int i;
	CUresult res;
	CUfunction function;
	CUmodule module;
	CUdeviceptr a_dev, b_dev, c_dev, n_dev;
	unsigned int *a = (unsigned int *) malloc (n*n * sizeof(unsigned int));
	unsigned int *b = (unsigned int *) malloc (n*n * sizeof(unsigned int));
	unsigned int *c = (unsigned int *) malloc (n*n * sizeof(unsigned int));

	printf("N = %d\n", n);
	for (i = 0; i < n*n; i++) {
		a[i] = i;
		c[i] = 0xff;
		if (i % (n + 1))
			b[i] = 0;
		else
			b[i] = 2;
	}

	res = cuModuleLoad(&module, "./matrixmul_gpu.cubin");
	if (res != CUDA_SUCCESS) {
		printf("cuModuleLoad() failed\n");
		return 0;
	}
	res = cuModuleGetFunction(&function, module, "_Z8multiplyPjS_S_S_");
	if (res != CUDA_SUCCESS) {
		printf("cuModuleGetFunction() failed\n");
		return 0;
	}
	
	cuFuncSetSharedSize(function, 0x40);
	cuFuncSetBlockShape(function, n, 1, 1);
	
	cuMemAlloc(&a_dev, n*n * sizeof(unsigned int));
	cuMemcpyHtoD(a_dev, a, n*n * sizeof(unsigned int));
	if (res != CUDA_SUCCESS) 
		printf("cuMemcpyHtoD (1) failed: res = %u\n", res);
	
	cuMemAlloc(&b_dev, n*n * sizeof(unsigned int));
	cuMemcpyHtoD(b_dev, b, n*n * sizeof(unsigned int));
	if (res != CUDA_SUCCESS) 
		printf("cuMemcpyHtoD (2) failed: res = %u\n", res);
	
	cuMemAlloc(&c_dev, n*n * sizeof(unsigned int));
	cuMemcpyHtoD(c_dev, c, n*n * sizeof(unsigned int));
	if (res != CUDA_SUCCESS) 
		printf("cuMemcpyHtoD (3) failed: res = %u\n", res);

	cuMemAlloc(&n_dev, sizeof(unsigned int));
	cuMemcpyHtoD(n_dev, &n, sizeof(unsigned int));
	if (res != CUDA_SUCCESS) 
		printf("cuMemcpyHtoD (4) failed: res = %u\n", res);

	cuParamSeti(function, 0, a_dev);	
	cuParamSeti(function, 4, a_dev >> 32);
	cuParamSeti(function, 8, b_dev);
	cuParamSeti(function, 12, b_dev >> 32);
	cuParamSeti(function, 16, c_dev);
	cuParamSeti(function, 20, c_dev >> 32);
	cuParamSeti(function, 24, n_dev);
	cuParamSeti(function, 28, n_dev >> 32);
	cuParamSetSize(function, 32);

	res = cuLaunchGrid(function, n, 1);
	if (res != CUDA_SUCCESS) 
		printf("cuLaunchGrid failed: res = %u\n", res);

	cuMemcpyDtoH(a, a_dev, n*n * sizeof(unsigned int));
	if (res != CUDA_SUCCESS) 
		printf("cuMemcpyDtoH (1) failed: res = %u\n", res);
	printf("A = ");
	for (i = 0; i < n*n; i++) {
		if (i % n == 0 && i != 0)
			printf(" ");
		printf("%d,", a[i]);
	}
	printf("\n");
	cuMemcpyDtoH(b, b_dev, n*n * sizeof(unsigned int));
	if (res != CUDA_SUCCESS) 
		printf("cuMemcpyDtoH (2) failed: res = %u\n", res);
	printf("B = ");
	for (i = 0; i < n*n; i++) {
		if (i % n == 0 && i != 0)
			printf(" ");
		printf("%d,", b[i]);
	}
	printf("\n");
	cuMemcpyDtoH(c, c_dev, n*n * sizeof(unsigned int));
	if (res != CUDA_SUCCESS) 
		printf("cuMemcpyDtoH (3) failed: res = %u\n", res);
	printf("C = ");
	for (i = 0; i < n*n; i++) {
		if (i % n == 0 && i != 0)
			printf(" ");
		printf("%d,", c[i]);
	}
	printf("\n");

	cuMemFree(a_dev);
	cuMemFree(b_dev);
	cuMemFree(c_dev);
	cuMemFree(n_dev);

	cuModuleUnload(module);

	free(a);
	free(b);
	free(c);

	return 0;
}

int main(int argc, char *argv[])
{
	CUresult res;
	CUdevice dev;
	CUcontext ctx;
	CUevent event[2];

	int count;
	size_t vram;
	int major, minor;

	unsigned int n = 3;

	if (argc > 1)
		n = atoi(argv[1]);

	res = cuInit(0);
	if (res != CUDA_SUCCESS) {
		printf("cuInit failed: res = %u\n", res);
		return 0;
	}

	res = cuDeviceGet(&dev, 0);
	if (res != CUDA_SUCCESS) {
		printf("cuDeviceGet failed: res = %u\n", res);
		return 0;
	}

	res = cuCtxCreate(&ctx, 0, dev);
	if (res != CUDA_SUCCESS) {
		printf("cuCtxCreate failed: res = %u\n", res);
		return 0;
	}

	test_matrixmul(n);

	res = cuCtxDestroy(ctx);
	if (res != CUDA_SUCCESS) {
		printf("cuCtxDestroy failed: res = %u\n", res);
		return 0;
	}
	
	printf("test finished\n");
	
	return 0;
}
