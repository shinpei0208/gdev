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

int cuda_test_memcpy_2step(unsigned int size)
{
	int i;
	CUresult res;
	CUdevice dev;
	CUcontext ctx;
	CUdeviceptr data_addr, data_addr2;
	unsigned int *in, *out;
	struct timeval tv;
	struct timeval tv_h2d_start, tv_h2d_end;
	float h2d;
	struct timeval tv_d2h_start, tv_d2h_end;
	float d2h;
	struct timeval tv_d2d_start, tv_d2d_end;
	float d2d;

	in = (unsigned int *) malloc(size);
	out = (unsigned int *) malloc(size);
	for (i = 0; i < size / 4; i++) {
		in[i] = i+1;
		out[i] = 0;
	}
	
	res = cuInit(0);
	if (res != CUDA_SUCCESS) {
		printf("cuInit failed: res = %u\n", (unsigned int)res);
		return -1;
	}

	res = cuDeviceGet(&dev, 0);
	if (res != CUDA_SUCCESS) {
		printf("cuDeviceGet failed: res = %u\n", (unsigned int)res);
		return -1;
	}

	res = cuCtxCreate(&ctx, 0, dev);
	if (res != CUDA_SUCCESS) {
		printf("cuCtxCreate failed: res = %u\n", (unsigned int)res);
		return -1;
	}

	res = cuMemAlloc(&data_addr, size);
	if (res != CUDA_SUCCESS) {
		printf("cuMemAlloc failed: res = %u\n", (unsigned int)res);
		return -1;
	}

	res = cuMemAlloc(&data_addr2, size);
	if (res != CUDA_SUCCESS) {
		printf("cuMemAlloc failed: res = %u\n", (unsigned int)res);
		return -1;
	}

	gettimeofday(&tv_h2d_start, NULL);
	res = cuMemcpyHtoD(data_addr, in, size);
	gettimeofday(&tv_h2d_end, NULL);
	if (res != CUDA_SUCCESS) {
		printf("cuMemcpyHtoD failed: res = %u\n", (unsigned int)res);
		return -1;
	}

	gettimeofday(&tv_d2d_start, NULL);
	res = cuMemcpyDtoD(data_addr2, data_addr, size);
	gettimeofday(&tv_d2d_end, NULL);
	if (res != CUDA_SUCCESS) {
		printf("cuMemcpyDtoD failed: res = %u\n", (unsigned int)res);
		return -1;
	}

	gettimeofday(&tv_d2h_start, NULL);
	res = cuMemcpyDtoH(out, data_addr2, size);
	gettimeofday(&tv_d2h_end, NULL);
	if (res != CUDA_SUCCESS) {
		printf("cuMemcpyDtoH failed: res = %u\n", (unsigned int)res);
		return -1;
	}

	res = cuMemFree(data_addr2);
	if (res != CUDA_SUCCESS) {
		printf("cuMemFree failed: res = %u\n", (unsigned int)res);
		return -1;
	}

	res = cuMemFree(data_addr);
	if (res != CUDA_SUCCESS) {
		printf("cuMemFree failed: res = %u\n", (unsigned int)res);
		return -1;
	}

	res = cuCtxDestroy(ctx);
	if (res != CUDA_SUCCESS) {
		printf("cuCtxDestroy failed: res = %u\n", (unsigned int)res);
		return -1;
	}

	for (i = 0; i < size / 4; i++) {
		if (in[i] != out[i]) {
			printf("in[%d] = %u, out[%d] = %u\n",
				   i, in[i], i, out[i]);
			goto end;
		}
	}

	free(in);
	free(out);

	tvsub(&tv_h2d_end, &tv_h2d_start, &tv);
	h2d = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
	tvsub(&tv_d2h_end, &tv_d2h_start, &tv);
	d2h = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
	tvsub(&tv_d2d_end, &tv_d2d_start, &tv);
	d2d = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;

	printf("HtoD: %f\n", h2d);
	printf("DtoD: %f\n", d2d);
	printf("DtoH: %f\n", d2h);

	return 0;

end:
	free(in);
	free(out);

	return -1;
}
